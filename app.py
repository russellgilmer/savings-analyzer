import streamlit as st
import pdfplumber
import pandas as pd
import re
from fuzzywuzzy import process
from openai import OpenAI

st.set_page_config(page_title="Merchant Statement Savings Analyzer", layout="wide")

# ===============================
# RATE SHEET LOADING & CLEANING
# ===============================
@st.cache_data
def load_clean_rate_sheet():
    xls = pd.ExcelFile("Curbstone Statement Template.xlsx")
    df_rates = pd.read_excel(xls, sheet_name="Interchange Savings")

    # Start at valid rows
    df_clean = df_rates.iloc[12:].copy()
    df_clean.columns = [
        "Category", "Volume", "Current Fee", "Current Cost",
        "Optimized Fee", "Adjusted Cost", "Savings", "Optimization Type"
    ]

    df_clean = df_clean.dropna(subset=["Category"])

    # Filter out notes/footers
    invalid_patterns = ["Effective", "Downgrade", "Categories that", "Depending on", "***", "**", "*"]
    def is_valid_category(cat):
        for pat in invalid_patterns:
            if pat.lower() in str(cat).lower():
                return False
        return True

    df_valid = df_clean[df_clean["Category"].apply(is_valid_category)].copy()
    df_valid = df_valid[["Category", "Current Fee", "Optimized Fee"]]

    # Normalize numeric
    df_valid["Current Fee"] = pd.to_numeric(df_valid["Current Fee"], errors="coerce")
    df_valid["Optimized Fee"] = pd.to_numeric(df_valid["Optimized Fee"], errors="coerce")
    return df_valid

rate_sheet = load_clean_rate_sheet()

# ===============================
# HELPER: RATE NORMALIZATION
# ===============================
def normalize_rate(val):
    """Ensure rates are in decimal form (0.025 not 2.5)."""
    if val is None:
        return 0.0
    try:
        v = float(val)
        return v / 100 if v > 1 else v
    except:
        return 0.0

# ===============================
# PDF PARSING (STRICT)
# ===============================
CATEGORY_KEYWORDS = ["PURCHAS", "CORPORATE", "BUSINESS", "TIER", "LEVEL", "LVL", "DATA"]

def extract_categories_from_pdf(pdf_file):
    """Extract ONLY valid Visa/MC category lines with rates."""
    extracted = []
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue
            for line in text.split("\n"):
                # Must have VS/VI/MC AND known keyword AND a % or decimal rate
                if re.search(r"(VS|VI|MC)", line.upper()) and \
                   any(k in line.upper() for k in CATEGORY_KEYWORDS) and \
                   re.search(r"\d+\.\d{2}", line):
                    extracted.append(line.strip())
    return list(set(extracted))  # unique valid lines only

def parse_line_to_components(line):
    """Extract category text, volume, and normalized rate."""
    # Category text before first $
    category = line.split("$")[0].strip()

    # Volume
    volume_match = re.findall(r"\$[\d,]+\.\d{2}", line)
    volume = float(volume_match[0].replace("$","").replace(",","")) if volume_match else 0.0

    # Rate (normalize)
    rate_match = re.findall(r"\d+\.\d{2,4}", line)
    stmt_rate = normalize_rate(rate_match[-1]) if rate_match else 0.0

    return category, volume, stmt_rate

# ===============================
# GPT FINAL REASONING
# ===============================
def gpt_choose_best(statement_category, statement_rate, candidates, api_key):
    client = OpenAI(api_key=api_key)
    candidate_text = "\n".join(
        f"- {c['Category']} (Current Fee: {normalize_rate(c['Current Fee']):.4%}, Optimized: {normalize_rate(c['Optimized Fee']):.4%})"
        for _, c in candidates.iterrows()
    )

    prompt = f"""
    You are a payment interchange expert.

    Statement category:
    "{statement_category}"
    Statement rate: {statement_rate:.4%}

    Possible matches:
    {candidate_text}

    Choose the best match considering:
    - Visa vs MC
    - Keyword similarity (Purchasing ≈ Purchasing, Level 2 ≈ LVL II)
    - Rate similarity

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
            {"role": "system", "content": "You are a payment interchange expert that maps statement categories to standardized interchange categories."},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content

# ===============================
# MATCHING PIPELINE
# ===============================
def match_and_calculate(lines, df_clean, api_key):
    results = []

    for line in lines:
        cat_text, volume, stmt_rate = parse_line_to_components(line)
        if stmt_rate == 0 or not cat_text:
            continue

        # Detect network
        network = "Visa" if "VS" in cat_text.upper() or "VI" in cat_text.upper() else "Mastercard" if "MC" in cat_text.upper() else "Other"

        # Step 1: Network filter
        if network == "Visa":
            net_df = df_clean[df_clean["Category"].str.contains("VI|Visa", case=False)]
        elif network == "Mastercard":
            net_df = df_clean[df_clean["Category"].str.contains("MC|Master", case=False)]
        else:
            net_df = df_clean

        # Step 2: Keyword filter
        keyword_hits = [k for k in CATEGORY_KEYWORDS if k in cat_text.upper()]
        key_df = net_df.copy()
        if keyword_hits:
            key_df = key_df[key_df["Category"].str.contains("|".join(keyword_hits), case=False)]

        # Step 3: Rate sanity filter (within 0.5%)
        rate_df = key_df.copy()
        if stmt_rate > 0:
            rate_df["diff"] = abs(normalize_rate(rate_df["Current Fee"]) - stmt_rate)
            rate_df = rate_df[rate_df["diff"] <= 0.005]

        # Candidate pool
        candidate_pool = rate_df if not rate_df.empty else key_df if not key_df.empty else net_df

        # Step 4: Decide match
        if len(candidate_pool) == 1:
            row = candidate_pool.iloc[0]
            matched_cat = row["Category"]
            opt_rate = normalize_rate(row["Optimized Fee"])
            curr_fee_ref = normalize_rate(row["Current Fee"])
            confidence = "High (Unique Filtered)"
        elif 1 < len(candidate_pool) <= 5:
            gpt_raw = gpt_choose_best(cat_text, stmt_rate, candidate_pool, api_key)
            try:
                import json
                gpt_res = json.loads(gpt_raw)
                matched_cat = gpt_res["best_match"]
                confidence = f"GPT {gpt_res.get('confidence','?')}%"
                row = candidate_pool[candidate_pool["Category"] == matched_cat].iloc[0]
                opt_rate = normalize_rate(row["Optimized Fee"])
                curr_fee_ref = normalize_rate(row["Current Fee"])
            except:
                best_fuzzy = process.extractOne(cat_text, candidate_pool["Category"].tolist())[0]
                row = candidate_pool[candidate_pool["Category"] == best_fuzzy].iloc[0]
                matched_cat = best_fuzzy
                opt_rate = normalize_rate(row["Optimized Fee"])
                curr_fee_ref = normalize_rate(row["Current Fee"])
                confidence = "Medium (Fuzzy Fallback)"
        else:
            matched_cat = None
            opt_rate = 0.0
            curr_fee_ref = 0.0
            confidence = "Unmatched"

        savings = volume * (stmt_rate - opt_rate) if matched_cat else 0.0

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
**How it works:**
1. Extracts only valid Visa/MC interchange lines (skips dates & totals)
2. Normalizes rates (1.9% = 0.019)
3. Filters by network, keywords, and rate similarity
4. Uses GPT ONLY for final ambiguous matches
5. Calculates potential savings
""")

api_key = st.text_input("Enter your OpenAI API Key", type="password")
uploaded_pdf = st.file_uploader("Upload Merchant Statement PDF", type=["pdf"])

if uploaded_pdf and api_key:
    with st.spinner("Extracting valid statement lines..."):
        categories_extracted = extract_categories_from_pdf(uploaded_pdf)
        st.success(f"✅ Found {len(categories_extracted)} valid category lines")

    if st.button("Run Analysis"):
        with st.spinner("Filtering → GPT reasoning → Calculating savings..."):
            savings_df = match_and_calculate(categories_extracted[:10], rate_sheet, api_key)
            total_savings = savings_df["Monthly Savings"].sum()

            st.subheader("📊 Savings Summary")
            st.dataframe(savings_df)

            st.markdown(f"**💰 Total Potential Monthly Savings: ${total_savings:,.2f}**")

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
