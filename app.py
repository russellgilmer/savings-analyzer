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
# GPT CATEGORY MATCHING
# ===============================
def gpt_match_category(statement_category, known_categories, api_key):
    """Use GPT to match a single statement category to the closest known category."""
    client = OpenAI(api_key=api_key)  # ✅ Create client *inside* the function

    prompt = f"""
    You are a payment interchange expert.

    Statement category from a merchant statement: "{statement_category}"

    Standard interchange categories: {known_categories}

    Find the best logical match from the list. Respond ONLY as JSON like:
    {{
      "statement": "...",
      "match": "...",
      "confidence": 0-100,
      "reason": "..."
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
# MATCH + VALIDATE + CALCULATE
# ===============================
def match_and_calculate(categories_extracted, df_clean, api_key):
    """For each parsed line, GPT matches it, then we calculate savings & confidence."""
    known_categories = df_clean["Category"].dropna().tolist()
    results = []

    for line in categories_extracted:
        cat, volume, stmt_rate = parse_line_to_components(line)

        # GPT match
        gpt_result_raw = gpt_match_category(cat, known_categories, api_key)

        # Try parsing GPT JSON
        try:
            import json
            gpt_result = json.loads(gpt_result_raw)
            matched_cat = gpt_result.get("match")
        except:
            # Fallback: fuzzy match if GPT JSON fails
            matched_cat = process.extractOne(cat, known_categories)[0]

        # Pull Current Fee & Optimized Fee for matched category
        opt_row = df_clean[df_clean["Category"] == matched_cat]
        if not opt_row.empty:
            current_fee_ref = float(opt_row["Current Fee"].values[0])
            opt_rate = float(opt_row["Optimized Fee"].values[0])
        else:
            current_fee_ref = 0.0
            opt_rate = 0.019  # default fallback

        # Confidence check: compare statement rate vs reference current fee
        if current_fee_ref > 0:
            rate_diff = abs(stmt_rate - current_fee_ref)
            if rate_diff <= 0.003:  # within ~0.3%
                confidence = "High"
            else:
                confidence = "Low (Rate mismatch)"
        else:
            confidence = "Unknown"

        # Calculate savings
        savings = volume * (stmt_rate - opt_rate)

        results.append({
            "Statement Category": cat,
            "Matched": matched_cat,
            "Statement Rate": stmt_rate,
            "Ref Current Fee": current_fee_ref,
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
The tool will extract interchange categories, match them to your standard categories, validate by rate, and calculate potential savings.
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

    # Step 2: Run GPT matching & savings
    if st.button("Run Analysis"):
        with st.spinner("Matching categories with GPT, validating by rate & calculating savings..."):
            # Limit to first 10 for demo speed (remove limit for full processing)
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
st.caption("Powered by Curbstone + GPT reasoning for interchange optimization")
