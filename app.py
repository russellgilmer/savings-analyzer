# save as app.py

import streamlit as st
import pdfplumber
import pandas as pd
import re
from fuzzywuzzy import process
import openai

# ===============================
# CONFIG
# ===============================
st.set_page_config(page_title="Merchant Statement Savings Analyzer", layout="wide")

# Load your spreadsheet once
@st.cache_data
def load_rate_sheet():
    xls = pd.ExcelFile("Curbstone Statement Template.xlsx")
    df_rates = pd.read_excel(xls, sheet_name="Interchange Savings")
    df_clean = df_rates.iloc[12:].copy()
    df_clean.columns = [
        "Category", "Volume", "Current Fee", "Current Cost",
        "Optimized Fee", "Adjusted Cost", "Savings", "Optimization Type"
    ]
    df_clean = df_clean.dropna(subset=["Category"])
    return df_clean

rate_sheet = load_rate_sheet()

# ===============================
# HELPER: PDF PARSER
# ===============================
def extract_categories_from_pdf(pdf_file):
    """Extracts statement categories, volumes & statement rates from PDF text"""
    extracted = []
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            for line in text.split("\n"):
                # Look for lines with a % rate + $ volume
                if re.search(r"\d+\.\d{2}\s", line) and "$" in line:
                    extracted.append(line.strip())
    # Deduplicate
    return list(set(extracted))

def parse_line_to_components(line):
    """Basic parser: extract category text, volume & rate"""
    # Try to isolate category name before $
    parts = line.split("$")
    category = parts[0].strip()
    # Find volume ($xxx.xx)
    volume_match = re.findall(r"\$[\d,]+\.\d{2}", line)
    volume = float(volume_match[0].replace("$","").replace(",","")) if volume_match else 0.0
    # Find rate (like 2.2000)
    rate_match = re.findall(r"\d\.\d{3,4}", line)
    rate = float(rate_match[-1]) if rate_match else 0.0
    return category, volume, rate

# ===============================
# HELPER: GPT CATEGORY MATCHING
# ===============================
def gpt_match_category(statement_category, known_categories):
    """Use GPT to match a single statement category"""
    prompt = f"""
    You are a payment interchange expert. 
    
    Statement category from a merchant statement: "{statement_category}"
    
    Standard interchange categories: {known_categories}
    
    Find the best logical match from the list. Respond as JSON with:
    {{
      "statement": "...",
      "match": "...",
      "confidence": 0-100,
      "reason": "..."
    }}
    """
    completion = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a payment interchange expert that matches merchant statement categories to standardized interchange categories."},
            {"role": "user", "content": prompt}
        ]
    )
    try:
        # Try to parse JSON from response
        return completion.choices[0].message.content
    except:
        return {"statement": statement_category, "match": None, "confidence": 0, "reason": "Failed to parse"}

def match_and_calculate(categories_extracted, df_clean):
    """For each parsed line, get GPT match, optimized rate, and savings"""
    known_categories = df_clean["Category"].dropna().tolist()
    results = []
    
    for line in categories_extracted:
        cat, volume, stmt_rate = parse_line_to_components(line)
        # Get GPT best match
        gpt_result_raw = gpt_match_category(cat, known_categories)
        # Attempt to parse GPT JSON result
        try:
            import json
            gpt_result = json.loads(gpt_result_raw)
            matched_cat = gpt_result.get("match")
        except:
            matched_cat = process.extractOne(cat, known_categories)[0]  # fallback fuzzy
        
        # Find optimized rate
        opt_row = df_clean[df_clean["Category"] == matched_cat]
        opt_rate = float(opt_row["Optimized Fee"].values[0]) if not opt_row.empty else 0.019
        
        savings = volume * (stmt_rate - opt_rate)
        results.append({
            "Statement Category": cat,
            "Matched": matched_cat,
            "Volume": volume,
            "Statement Rate": stmt_rate,
            "Optimized Rate": opt_rate,
            "Monthly Savings": savings
        })
    return pd.DataFrame(results)

# ===============================
# STREAMLIT UI
# ===============================
st.title("💳 Merchant Statement Interchange Savings Analyzer")

st.markdown("Upload a merchant credit card statement PDF. The tool will extract interchange categories, match them to your standard categories, and calculate potential savings.")

api_key = st.text_input("Enter your OpenAI API Key", type="password")
if api_key:
    openai.api_key = api_key

uploaded_pdf = st.file_uploader("Upload Merchant Statement PDF", type=["pdf"])

if uploaded_pdf and api_key:
    with st.spinner("Extracting categories..."):
        categories_extracted = extract_categories_from_pdf(uploaded_pdf)
        st.success(f"Extracted {len(categories_extracted)} category lines")

    if st.button("Run Analysis"):
        with st.spinner("Matching categories and calculating savings..."):
            savings_df = match_and_calculate(categories_extracted[:10], rate_sheet)  # limit 10 for speed
            total_savings = savings_df["Monthly Savings"].sum()

            st.subheader("📊 Savings Summary")
            st.dataframe(savings_df)

            st.markdown(f"**Total Potential Monthly Savings: ${total_savings:,.2f}**")

            # Allow CSV download
            csv = savings_df.to_csv(index=False)
            st.download_button(
                "📥 Download Savings Report",
                csv,
                "savings_report.csv",
                "text/csv"
            )
else:
    st.info("Upload a PDF and enter your OpenAI API key to start.")

st.markdown("---")
st.caption("Powered by Curbstone + GPT reasoning for category optimization")
