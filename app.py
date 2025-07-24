import streamlit as st
import pdfplumber
import pandas as pd
import re
from fuzzywuzzy import process
from openai import OpenAI

# ===============================
# STREAMLIT CONFIG
# ===============================
st.set_page_config(page_title="Merchant Statement Savings Analyzer", layout="wide")

# ===============================
# LOAD & CLEAN RATE SHEET
# ===============================
@st.cache_data
def load_clean_rate_sheet():
    """Load the Curbstone template, clean out notes, keep only valid categories with both fees."""
    xls = pd.ExcelFile("Curbstone Statement Template.xlsx")
    df_rates = pd.read_excel(xls, sheet_name="Interchange Savings")

    # Start at valid rows
    df_clean = df_rates.iloc[12:].copy()
    df_clean.columns = [
        "Category", "Volume", "Current Fee", "Current Cost",
        "Optimized Fee", "Adjusted Cost", "Savings", "Optimization Type"
    ]

    # Drop empty rows
    df_clean = df_clean.dropna(subset=["Category"])

    # Filter only valid categories (remove notes/explanations)
    invalid_patterns = ["Effective", "Downgrade", "Categories that", "Depending on", "***", "**", "*"]
    def is_valid_category(cat):
        if pd.isna(cat):
            return False
        for pat in invalid_patterns:
            if pat.lower() in str(cat).lower():
                return False
        return True

    df_valid_full = df_clean[df_clean["Category"].apply(is_valid_category)].copy()

    # Keep only necessary columns
    df_valid_trimmed = df_valid_full[["Category", "Current Fee", "Optimized Fee"]].copy()

    # Convert to numeric
    df_valid_trimmed["Current Fee"] = pd.to_numeric(df_valid_trimmed["Current Fee"], errors="coerce")
    df_valid_trimmed["Optimized Fee"] = pd.to_numeric(df_valid_trimmed["Optimized Fee"], errors="coerce")

    return df_valid_trimmed

rate_sheet = load_clean_rate_sheet()

# ===============================
# PDF PARSING
# ===============================
def extract_categories_from_pdf(pdf_file):
    """Extracts statement categories, volumes & statement rates from PDF text."""
    extracted = []
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                for line in text.split("\n"):
                    # Look for lines with % rate and $
                    if re.search(r"\d+\.\d{2}\s", line) and "$" in line:
                        extracted.append(line.strip())
    return list(set(extracted))  # remove duplicates

def parse_line_to_components(line):
    """Extract category, volume & rate from a single line."""
    # Take text before first $
    parts = line.split("$")
    category = parts[0].strip()

    # Volume (first $amount in line)
    volume_match = re.findall(r"\$[\d,]+\.\d{2}", line)
    volume = float(volume_match[0].replace("$","").replace(",","")) if volume_match else 0.0

    # Rate (like 2.2000)
    rate_match = re.findall(r"\d\.\d{3,4}", line)
    rate = float(rate_match[-1]) if rate_match else 0.0

    return category, volume, rate

# ===============================
# GPT REASONING (LAST STEP)
# ===============================
def gpt_choose_best(statement_category, statement_rate, candidates, api_key):
    """Ask GPT to choose the best match from a small filtered list."""
    client = OpenAI(api_key=api_key)

    candidate_text = "\n".join(
        f"- {c['Category']} (Current Fee: {c['Current Fee']:.4%}, Optimized: {c['Optimized Fee']:.4%})"
        for _, c in candidates.iterrows()
    )

    prompt = f"""
    You are a payment interchange expert.

    Statement category from merchant:
    - Description: "{statement_category}"
    - Rate on statement: {statement_rate:.4%}

    Here are possible standard interchange categories:
    {candidate_text}

    Which one best matches the statement category?
    Consider:
    - Keyword similarity (Purchasing ≈ Purchasing, Level 2 ≈ LVL II)
    - Network match (Visa vs MC)
    - Rate similarity (statement vs current fee)

    Respond ONLY as JSON:
    {{
      "best_match": "...",
      "reason": "...",
      "confidence": 0-100
    }}
    """

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a payment interchange expert that matches merchant statement categories to standardized interchange categories."},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content

# ===============================
# MULTI-STEP FILTER + MATCH + CALC
# ===============================
def match_and_calculate(categories_extracted, df_clean, api_key):
    """Strict multi-step matching before GPT reasoning."""
    results = []

    for line in categories_extracted:
        cat_text, volume, stmt_rate = parse_line_to_components(line)
        network = "Visa" if "VS" in cat_text.upper() else "Mastercard" if "MC" in cat_text.upper() else "Other"

        # Step 1: Filter by network
        network_filter = df_clean[df_clean["Category"].str.contains("VI|Visa", case=False) if network == "Visa"
                                 else df_clean["Category"].str.contains("MC|Mastercard", case=False)]

        # Step 2: Keyword filter (Purchasing, Tier, Level)
        keywords = []
        if "PURCH" in cat_text.upper():
            keywords.append("Purchasing")
        if "TIER" in cat_text.upper():
            keywords.append("Tier")
        if "LEVEL" in cat_text.upper() or "LVL" in cat_text.upper():
            keywords.append("Level")

        keyword_filtered = network_filter.copy()
        if keywords:
            keyword_filtered = keyword_filtered[keyword_filtered["Category"].str.contains("|".join(keywords), case=False)]

        # Step 3: Rate sanity filter (within ~0.5%)
        rate_filtered = keyword_filtered.copy()
        if stmt_rate > 0:
            rate_filtered["diff"] = abs(rate_filtered["Current Fee"] - stmt_rate)
            rate_filtered = rate_filtered[rate_filtered["diff"] <= 0.005]

        # If no match after filtering, fallback to full network list
        candidate_pool = rate_filtered if not rate_filtered.empty else keyword_filtered

        # Step 4: If only 1 candidate → auto match
        if len(candidate_pool) == 1:
            match_row = candidate_pool.iloc[0]
            matched_cat = match_row["Category"]
            opt_rate = match_row["Optimized Fee"]
            curr_fee_ref = match_row["Current Fee"]
            confidence = "High (Unique Filtered Match)"
        elif len(candidate_pool) > 1:
            # Step 5: GPT choose best from small list
            gpt_choice_raw = gpt_choose_best(cat_text, stmt_rate, candidate_pool.head(5), api_key)
            try:
                import json
                gpt_choice = json.loads(gpt_choice_raw)
                matched_cat = gpt_choice["best_match"]
                confidence = f"GPT {gpt_choice.get('confidence', '?')}%"
                # find rates
                match_row = candidate_pool[candidate_pool["Category"] == matched_cat].iloc[0]
                opt_rate = match_row["Optimized Fee"]
                curr_fee_ref = match_row["Current Fee"]
            except:
                # fallback to best fuzzy
                best_fuzzy = process.extractOne(cat_text, candidate_pool["Category"].tolist())[0]
                match_row = candidate_pool[candidate_pool["Category"] == best_fuzzy].iloc[0]
                matched_cat = best_fuzzy
                opt_rate = match_row["Optimized Fee"]
                curr_fee_ref = match_row["Current Fee"]
                confidence = "Medium (Fuzzy Fallback)"
        else:
            matched_cat = None
            opt_rate = 0.0
            curr_fee_ref = 0.0
            confidence = "Unmatched"

        # Savings calc
        savings = volume * (stmt_rate - opt_rate) if matched_cat else 0

        results.append({
            "Statement Category": cat_text,
            "Matched": matched_cat if matched_cat else "No Match",
            "Statement Rate": stmt_rate,
            "Ref Current Fee": curr_fee_ref,
            "Optimized Rate": opt_rate,
            "Volume": volume,
            "Monthly Savings": savings,
            "Confidence": confidence
        })

    return pd.DataFrame(results)

# ===============================
# STREAMLIT UI
# ===============================
st.title("💳 Merchant Statement Interchange Savings Analyzer")

st.markdown("""
Upload a merchant credit card statement PDF.  
The tool will:
1. Detect Visa vs MasterCard  
2. Pre-filter by keywords & rate similarity  
3. Let GPT choose only if needed  
4. Validate matches & calculate savings
""")

# OpenAI API key input
api_key = st.text_input("Enter your OpenAI API Key", type="password")

# File uploader
uploaded_pdf = st.file_uploader("Upload Merchant Statement PDF", type=["pdf"])

if uploaded_pdf and api_key:
    # Step 1: Extract categories
    with st.spinner("Extracting categories from statement..."):
        categories_extracted = extract_categories_from_pdf(uploaded_pdf)
        st.success(f"✅ Extracted {len(categories_extracted)} category lines from the statement.")

    # Step 2: Run strict matching & savings
    if st.button("Run Analysis"):
        with st.spinner("Filtering → GPT reasoning → Calculating savings..."):
            savings_df = match_and_calculate(categories_extracted[:10], rate_sheet, api_key)
            total_savings = savings_df["Monthly Savings"].sum()

            st.subheader("📊 Savings Summary")
            st.dataframe(savings_df)

            st.markdown(f"**💰 Total Potential Monthly Savings: ${total_savings:,.2f}**")

            # CSV download
            csv = savings_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Savings Report CSV",
                data=csv,
                file_name="savings_report.csv",
                mime="text/csv"
            )
else:
    st.info("⬆️ Upload a PDF and enter your OpenAI API key to start.")

st.markdown("---")
st.caption("Powered by Curbstone + strict filtering + GPT reasoning")
