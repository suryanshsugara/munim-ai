# app.py
import io
import json
import logging
import os
from typing import Any, Dict, List

import pandas as pd
import pdfplumber
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai

# ---------- Setup ----------

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

load_dotenv()

def get_gemini_api_key() -> str:
    """
    Try to read GEMINI_API_KEY from environment (local dev)
    or from Streamlit secrets (Streamlit Cloud).
    """
    key = os.getenv("GEMINI_API_KEY")

    # Streamlit Cloud: use st.secrets if env var not set
    if not key:
        try:
            key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            key = None

    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. "
            "Set it in a local .env file or in Streamlit secrets."
        )
    return key


st.set_page_config(
    page_title="Munim AI",
    page_icon="💰",
    layout="wide",
)


# ---------- PDF Utilities ----------

def extract_clean_lines_from_pdf(file_obj: io.BytesIO) -> List[str]:
    try:
        with pdfplumber.open(file_obj) as pdf:
            if not pdf.pages:
                raise ValueError("PDF has no pages.")

            pages_lines = []
            for page in pdf.pages:
                text = page.extract_text() or ""
                lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
                pages_lines.append(lines)

    except Exception as e:
        logging.exception("Failed to read/parse PDF.")
        raise RuntimeError(
            "Could not read this PDF. It may be scanned, password-protected, or corrupted."
        ) from e

    # Remove repeating headers/footers
    from collections import Counter
    freq = Counter()
    for page in pages_lines:
        freq.update(set(page))

    threshold = int(0.7 * len(pages_lines))
    cleaned = []
    for page in pages_lines:
        for ln in page:
            if freq[ln] < threshold:
                cleaned.append(ln)

    if not cleaned:
        raise RuntimeError("No useful text extracted from this PDF.")

    return cleaned


def extract_transaction_text(uploaded_file) -> str:
    uploaded_file.seek(0)
    data = io.BytesIO(uploaded_file.read())
    lines = extract_clean_lines_from_pdf(data)
    text = "\n".join(lines)

    # safety limit
    if len(text) > 15000:
        text = text[:15000]

    return text


# ---------- LLM Utilities (Gemini) ----------

def get_gemini_model():
    """Configure Gemini and pick a model that supports generateContent."""
    api_key = get_gemini_api_key()
    genai.configure(api_key=api_key)

    try:
        models = [
            m for m in genai.list_models()
            if "generateContent" in getattr(m, "supported_generation_methods", [])
        ]
    except Exception as e:
        raise RuntimeError(
            f"Could not list Gemini models. Check your API key / project. Raw error: {e}"
        )

    if not models:
        raise RuntimeError(
            "No Gemini models with generateContent are available for this API key."
        )

    preferred_order = [
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-1.0-pro",
        "gemini-pro",
    ]

    chosen_name = None
    for pref in preferred_order:
        for m in models:
            if pref in m.name:
                chosen_name = m.name
                break
        if chosen_name:
            break

    if not chosen_name:
        chosen_name = models[0].name

    logging.info(f"Using Gemini model: {chosen_name}")
    return genai.GenerativeModel(chosen_name)


def analyze_transactions_with_llm(transaction_text: str) -> Dict[str, Any]:
    model = get_gemini_model()

    system_prompt = """
You are 'Munim AI' — a sharp, witty, insightful Indian financial advisor.
Your personality is:
- Observant, analytical, and grounded in financial reality
- Lightly sarcastic, but never rude, bitter, parental, or moralizing
- Calm, confident, professional — like a consultant who secretly enjoys roasting bad money habits
- No emojis, no slang, no preachiness

Your task: Analyze the raw transaction data provided below.

1. Categorize spending:
   - Needs: rent, groceries, utilities, fuel, insurance, basic transport
   - Wants: food delivery, cafes, shopping, OTT, travel, events, impulse buys, lifestyle upgrades

2. Identify money leakages:
   - Recurring subscriptions the user may not need
   - Repeated late-night food orders or high-frequency small spends
   - Excessive UPI transfers to the same individuals or merchants
   - Irregular spending spikes

3. Deliver a roast:
   - 2–4 short paragraphs
   - Smart, financially grounded humor
   - Be specific — reference merchants, brands, timings, and patterns
   - Roast the spending behaviour, not the person
   - Maintain respect, avoid personal judgments

Output format (STRICT JSON only — no markdown, no commentary):

{
  "financial_score": 0-100 integer,
  "roast": "Insightful, witty critique of spending patterns.",
  "top_wasters": [
    {
      "category": "Food Delivery",
      "amount": "₹4,500",
      "comment": "Ordering convenience is costing more than cooking ever would."
    },
    {
      "category": "Subscriptions",
      "amount": "₹899",
      "comment": "Multiple OTT platforms, unclear usage — consider trimming."
    }
  ],
  "one_click_fix": "One immediate action that saves the most money."
}

Rules:
- Return ONLY valid JSON with double quotes.
- Do NOT wrap output in backticks or any other formatting.
- If the data is incomplete, acknowledge uncertainty in the roast.
"""


    full_prompt = f"""{system_prompt}

RAW_TRANSACTION_TEXT:
\"\"\"{transaction_text}\"\"\"
"""

    try:
        response = model.generate_content(
            full_prompt,
            generation_config={"temperature": 0.3},
        )

        content = response.text.strip()

        # In case the model still wraps in ```json ... ```
        if content.startswith("```"):
            content = content.strip("`")
            if content.lower().startswith("json"):
                content = content[4:].strip()

        data = json.loads(content)
        return data

    except Exception as e:
        logging.exception("LLM analysis failed.")
        raise RuntimeError(f"LLM error: {e}") from e



# ---------- UI ----------

def show_header():
    st.title("Munim AI")
    st.caption("Upload your bank statement and get a brutally honest spending review.")


def show_output(result: Dict[str, Any]):
    # Layout: score + quick summary on top
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Financial Score")
        score = result.get("financial_score", "N/A")
        if isinstance(score, (int, float)):
            score_int = max(0, min(100, int(score)))
            st.metric("Score (0–100)", score_int)
            st.progress(score_int / 100)
        else:
            st.metric("Score (0–100)", "N/A")

    with col2:
        st.subheader("One-Click Fix")
        fix = result.get("one_click_fix") or "No clear fix suggested."
        st.info(fix)

    st.markdown("---")

    st.subheader("Roast")
    st.write(result.get("roast", ""))

    st.markdown("---")

    st.subheader("Top Money Wasters")
    df = pd.DataFrame(result.get("top_wasters", []))

    if not df.empty:
        # Ensure column order and nicer names if present
        rename_map = {
            "category": "Category",
            "amount": "Amount",
            "comment": "Comment",
        }
        df = df.rename(columns=rename_map)
        st.dataframe(df, use_container_width=True)
    else:
        st.write("No obvious money-waster categories detected. Either you're a saint or the data is too thin.")


def main():
    show_header()

    with st.sidebar:
        st.markdown("### How it works")
        st.markdown(
            "- Upload a recent bank statement (PDF).\n"
            "- Munim AI parses your transactions.\n"
            "- You get a score, roast, and top money leaks."
        )

    uploaded_file = st.file_uploader(
        "Upload bank statement PDF",
        type=["pdf"],
        help="Export a normal text-based PDF from your bank (not a scanned image).",
    )

    if uploaded_file and st.button("Analyze"):
        with st.spinner("Reading your statement..."):
            try:
                tx_text = extract_transaction_text(uploaded_file)
            except Exception as e:
                st.error(str(e))
                return

        with st.spinner("Asking Munim AI to analyse your spending..."):
            try:
                result = analyze_transactions_with_llm(tx_text)
            except Exception as e:
                if "API key" in str(e):
                    st.error("API key issue. Check GEMINI_API_KEY in your .env file.")
                else:
                    st.error("AI analysis failed. Details below:")
                    st.exception(e)
                return

        show_output(result)


# ---------- Main ----------

def main():
    show_header()

    uploaded_file = st.file_uploader(
        "Upload bank statement PDF",
        type=["pdf"],
    )

    if uploaded_file and st.button("Analyze 🔍"):
        with st.spinner("Extracting transactions..."):
            try:
                tx_text = extract_transaction_text(uploaded_file)
            except Exception as e:
                st.error(str(e))
                return

        with st.spinner("Asking Munim AI to judge you..."):
            try:
                result = analyze_transactions_with_llm(tx_text)
            except Exception as e:
                if "API key" in str(e):
                    st.error("🔑 Gemini API key issue. Check GEMINI_API_KEY in your .env.")
                else:
                    st.error("AI analysis failed. Details below:")
                    st.exception(e)
                return

        show_output(result)


if __name__ == "__main__":
    main()
