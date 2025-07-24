import streamlit as st
import pdfplumber
import pandas as pd
import re
from fuzzywuzzy import process
from openai import OpenAI

st.set_page_config(page_title="Merchant Statement Savings Analyzer", layout="wide")

# ===============================
# LOAD & CLEAN RATE SHEET
# ===============================
@st.cache_data
def load_clean_rate_sheet():
    xls = pd.ExcelFile("Curbstone Statement Template.xlsx")
    df_rates = pd.read_excel(xls, sheet_name="Interchange Savings")

    df_clean = df_rates.iloc[12:].copy()
    df_clean.columns = [
        "Category", "Volume", "Current Fee", "Current Cost",
        "Optimized Fee", "Adjusted Cost", "Savings", "Optimization Type"
    ]
    df_clean = df_clean.dropna(subset=["Category"])

    invalid_patterns = ["Effective", "Downgrade", "Categories that", "Depending on", "***", "**", "*"]
    def is_valid_category(cat):
        for pat in invalid_patterns:
            if pat.lower() in str(cat).lower():
                return False
        return True

    df_valid = df_clean[df_clean["Category"].apply(is_valid_category)].copy()
    df_valid = df_valid[["Category", "Current Fee", "Optimized Fee"]]

    df_valid["Current Fee"] = pd.to_numeric(df_valid["Current Fee"], errors="coerce")
    df_valid["Optimized Fee"] = pd.to_numeric(df_valid["Optimized Fee"], errors="coerce")
    return df_valid

rate_sheet = load_clean_rate_sheet()

# ===============================
# HELPERS
# ===============================
def normalize_rate(val):
    if val is None:
        return 0.0
    try:
        v = float(val)
        return v / 100 if v > 1 else v
    except:
        return 0.0

def extract_network(text):
    if "VS" in text.upper() or "VI" in text.upper():
        return "Visa"
    elif "MC" in text.upper():
        return "Mastercard"
    return "Other"

# ===============================
# PDF EXTRACTION
# ===============================
def extract_statement_lines(pdf_file):
    valid_lines = []
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue
            for line in text.split("\n"):
                if re.search(r"(VS|VI|MC)", line.upper()) and re.search(r"\d+\.\d{1,4}", line):
                    valid_lines.append(line.strip())
    return list(set(valid_lines))

def parse_line(line):
    category_text = line.split("$")[0].strip()
    vol_match = re.findall(r"\$[\d,]+\.\d{2}", line)
    volume = float(vol_match[0].replace("$","").replace(",","")) if vol_match else 0.0

    rate_match = re.findall(r"\d+\.\d{1,4}", line)
    stmt_rate = 0.0
    if rate_match:
        stmt_rate = float(rate_match[-1])
        if stmt_rate > 1:
            stmt_rate = stmt_rate / 100
    return category_text, volume, stmt_rate

# ===============================
# GPT MATCHING
# ===============================
def gpt_reason_match(statement_category, statement_rate, network, candidate_rows, api_key):
    client = OpenAI(api_key=api_key)

    candidate_text = "\n".join(
        f"- {row['Category']} (Current Fee: {normalize_rate(row['Current Fee']):.4%}, Optimized: {normalize_rate(row['Optimized Fee']):.4%})"
        for _, row in candidate_rows.iterrows()
    )

    prompt = f"""
    You are a payment interchange expert.

    Merchant statement:
    - "{statement_category}"
    - Network: {network}
    - Discount Rate: {statement_rate:.4%}

    Possible matching {network} categories:
    {candidate_text}

    Reason like a human:
    - Compare keywords (Purchasing ≈ Purchasing, Level 2 ≈ LVL II)
    - Compare rates (2.5% = 0.025)
    - Pick the closest even if not exact
    - If truly no match, say "No Match"

    Respond in JSON:
    {{
      "best_match": "...",
      "reason": "...",
      "confidence": 0-100
    }}
    """

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a payment interchange expert mapping merchant statement categories to standard interchange categories."},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content

# ===============================
# HYBRID MATCH LOOP
# ===============================
def hybrid_match_and_calculate(lines, df_clean, api_key):
    results = []

    for line in lines:
        cat_text, volume, stmt_rate = parse_line(line)
        if stmt_rate == 0:
            continue

        # Detect network
        network = extract_network(cat_text)

        # Filter spreadsheet to network
        if network == "Visa":
            df_net = df_clean[df_clean["Category"].str.contains("VI|Visa", case=False)]
        elif network == "Mastercard":
            df_net = df_clean[df_clean["Category"].str.contains("MC|Master", case=False)]
        else:
            df_net = df_clean

        # Fuzzy pre-filter: find top 15 closest by name
        top_matches = []
        for cat in df_net["Category"].tolist():
            score = process.extractOne(cat_text, [cat])[1]
            top_matches.append((cat, score))
        top_matches = sorted(top_matches, key=lambda x: x[1], reverse=True)[:15]
        top_df = df_net[df_net["Category"].isin([m[0] for m in top_matches])]

        # GPT reasons among top 15
        gpt_raw = gpt_reason_match(cat_text, stmt_rate, network, top_df, api_key)

        try:
            import json
            gpt_res = json.loads(gpt_raw)
            matched_cat = gpt_res.get("best_match", "No Match")
            reason = gpt_res.get("reason", "")
            confidence = gpt_res.get("confidence", 0)
        except:
            matched_cat = "No Match"
            reason = "GPT parsing failed"
            confidence = 0

        opt_rate, curr_fee_ref, savings = 0.0, 0.0, 0.0
        if matched_cat and matched_cat != "No Match":
            match_row = df_net[df_net["Category"] == matched_cat]
            if not match_row.empty:
                curr_fee_ref = normalize_rate(match_row["Current Fee"].values[0])
                opt_rate = normalize_rate(match_row["Optimized Fee"].values[0])
                savings = volume * (stmt_rate - opt_rate)

        results.append({
            "Statement Category": cat_text,
            "Matched": matched_cat,
            "Statement Rate": stmt_rate,
            "Ref Current Fee": curr_fee_ref,
            "Optimized Rate": opt_rate,
            "Volume": volume,
            "Monthly Savings": savings,
            "Confidence": confidence,
            "Reason": reason
        })

    return pd.DataFrame(results)

# ===============================
# STREAMLIT UI
# ===============================
st.title("💳 Merchant Statement Analyzer (Hybrid Fuzzy + GPT Reasoning)")

st.markdown("""
**How this works:**
1. Extracts all Visa/MC lines with a discount %  
2. Fuzzy pre-filters top 15 most similar categories for that network  
3. GPT reasons like a human to pick the closest match (or No Match)  
4. GPT explains its reasoning  
5. Savings = Volume × (Statement Rate – Optimized Fee)
""")

api_key = st.text_input("Enter your OpenAI API Key", type="password")
uploaded_pdf = st.file_uploader("Upload Merchant Statement PDF", type=["pdf"])

if uploaded_pdf and api_key:
    with st.spinner("Extracting statement lines..."):
        statement_lines = extract_statement_lines(uploaded_pdf)
        st.success(f"✅ Found {len(statement_lines)} lines with VS/MC + discount rates")

    if st.button("Run Hybrid AI Analysis"):
        with st.spinner("Fuzzy filtering → GPT reasoning → Calculating savings..."):
            savings_df = hybrid_match_and_calculate(statement_lines[:10], rate_sheet, api_key)
            total_savings = savings_df["Monthly Savings"].sum()

            st.subheader("📊 GPT Hybrid Matches (with Reasoning)")
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
st.caption("Powered by Curbstone + Hybrid Fuzzy Filtering + GPT Reasoning")
