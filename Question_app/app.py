import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from PyPDF2 import PdfReader
import torch

# Load the model (use CUDA if available)
model_name = "google/flan-t5-base"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

def generate_questions(text, max_new_tokens=256):
    prompt = f"""
You are a senior compliance officer reviewing this document.
Based on the content, generate a list of due diligence questions to assess regulatory and legal risks.

Document:
{text}

Questions:
"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def extract_text_from_pdf(uploaded_file):
    pdf = PdfReader(uploaded_file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""
    return text

# UI
st.set_page_config(page_title="Due Diligence Generator", layout="centered")
st.title("📄 Due Diligence Question Generator")
st.markdown("Upload a PDF document to extract key due diligence questions.")

uploaded_file = st.file_uploader("Upload a document", type=["pdf"])

if uploaded_file:
    with st.spinner("🔍 Extracting text from document..."):
        text = extract_text_from_pdf(uploaded_file)

    if text:
        if st.button("🚀 Generate Questions"):
            with st.spinner("Generating questions..."):
                questions = generate_questions(text)
            st.markdown("### ✅ Generated Questions")
            st.write(questions)
    else:
        st.warning("No text could be extracted from the document.")
