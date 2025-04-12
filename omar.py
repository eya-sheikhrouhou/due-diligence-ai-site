import streamlit as st
from PyPDF2 import PdfReader
from red_flag_checker import get_flagged_sentences
from llm_risk_analyzer import analyze_risks_with_llm # type: ignore

# Config Streamlit
st.set_page_config("🚩 Smart Red Flag Detector", layout="centered")
st.markdown("<h1 style='text-align:center;'>🚩 Smart Red Flag Detector</h1>", unsafe_allow_html=True)

# Upload
uploaded_file = st.file_uploader("📎 Upload a PDF file", type="pdf")

if uploaded_file is not None:
    st.success("✅ File uploaded successfully!")

    # Extract text
    reader = PdfReader(uploaded_file)
    full_text = "\n".join([page.extract_text() or "" for page in reader.pages])

    # --- REGEX DETECTION ---
    if st.button("🔍 Scan for Red Flags"):
        flagged = get_flagged_sentences(full_text)

        if flagged:
            st.error(f"🚨 {len(flagged)} red flag(s) detected in this document.")
            
            for item in flagged:
                color = {
                    "high": "#e74c3c",
                    "medium": "#f39c12",
                    "low": "#2ecc71"
                }[item["severity"]]

                st.markdown(f"""
                    <div style='
                        background-color: #1e1e1e;
                        border-left: 6px solid {color};
                        padding: 15px;
                        margin-bottom: 20px;
                        border-radius: 8px;
                        font-family: monospace;
                        line-height: 1.5;
                        color: #f0f0f0;
                    '>
                        <p><strong>Severity:</strong> <span style='color:{color};'>{item['severity'].capitalize()}</span></p>
                        <p><strong>Pattern:</strong> <code>{item['pattern']}</code></p>
                        <p><strong>Context:</strong> {item['sentence']}</p>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.success("✅ No red flags found. All clear!")

    # --- LLM-BASED RISK ANALYSIS ---
    if st.button("🤖 Analyze with LLM"):
        with st.spinner("Generating AI-powered risk report..."):
            ai_risks = analyze_risks_with_llm(full_text)

        formatted_risks = ai_risks.replace("\n", "<br>")
    
        st.subheader("🤖 LLM-Based Risk Analysis")
        st.markdown(f"""
            <div style='background-color:#111; padding:15px; border-radius:8px; color:white; font-family:monospace'>
            {formatted_risks}
            </div>
        """, unsafe_allow_html=True)

