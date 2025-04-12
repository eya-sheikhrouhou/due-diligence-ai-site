from flask import Flask, render_template_string, request
from transformers import pipeline
from PyPDF2 import PdfReader
import re

# --- Red Flag Patterns ---
RED_FLAGS = {
    r"\bfraud(ulent|s|)\b": "high",
    r"\bunder investigation\b": "high",
    r"\bpending litigation\b": "medium",
    r"\bbreach of contract\b": "medium",
    r"\bterminated.*CEO\b": "low",
    r"\bbankrupt(cy|)\b": "high",
    r"\bSEC (notice|action)\b": "medium",
    r"\bsanction(ed)?\b": "medium",
    r"\bcriminal charge(s)?\b": "high",
    r"\bloss of customer data\b": "high"
}

def get_flagged_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    results = []
    for sent in sentences:
        for pattern, severity in RED_FLAGS.items():
            match = re.search(pattern, sent, re.IGNORECASE)
            if match:
                highlighted = re.sub(
                    pattern,
                    f"<strong style='color:#e74c3c;'>" + match.group(0) + "</strong>",
                    sent,
                    flags=re.IGNORECASE
                )
                results.append({
                    "sentence": highlighted.strip(),
                    "pattern": pattern,
                    "severity": severity
                })
    return results

# --- Flask App Setup ---
app = Flask(__name__, static_url_path='', static_folder='.')
question_generator = pipeline("text2text-generation", model="google/flan-t5-base", device=-1)

with open("index.html", encoding="utf-8") as f:
    html_template = f.read()

@app.route("/", methods=["GET", "POST"])
def index():
    questions = []
    red_flags = []

    if request.method == "POST":
        uploaded_file = request.files.get("pdf")
        action = request.form.get("action")

        if uploaded_file and uploaded_file.filename.endswith(".pdf"):
            # Extract text from PDF
            text = ""
            reader = PdfReader(uploaded_file)
            for page in reader.pages:
                content = page.extract_text()
                if content:
                    text += content.strip() + "\n"

            if action == "red_flags":
                red_flags = get_flagged_sentences(text)

            elif action == "generate_questions":
                prompt = f"""
You are an AI assistant helping with due diligence analysis.
Based on the following text, generate exactly 5 clear and diverse questions.
Each question should:
- Start on a new line
- Begin with What, Who, When, or How (avoid Why)
- Be phrased concisely
Format: one question per line, numbered.

Text:
{text[:1500]}
"""
                response = question_generator(prompt, max_length=512, do_sample=False)[0]["generated_text"]
                questions = re.findall(r"(?:^|\n)\s*\d[\.\)]\s*(.+)", response)

                if not questions:
                    questions = [q.strip() for q in response.split("\n") if q.strip()]

    return render_template_string(html_template, questions=questions, red_flags=red_flags)

if __name__ == "__main__":
    app.run(debug=True)