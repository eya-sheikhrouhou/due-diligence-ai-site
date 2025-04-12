from flask import Flask, render_template_string, request
from transformers import pipeline
from PyPDF2 import PdfReader
import re

app = Flask(__name__, static_url_path='', static_folder='.')  # important for root index.html

# Load the model
question_generator = pipeline("text2text-generation", model="google/flan-t5-base", device=-1)

# Load the HTML template
with open("index.html", encoding="utf-8") as f:
    html_template = f.read()

@app.route("/", methods=["GET", "POST"])
def index():
    questions = []
    if request.method == "POST":
        uploaded_file = request.files["pdf"]
        if uploaded_file and uploaded_file.filename.endswith(".pdf"):
            text = ""
            reader = PdfReader(uploaded_file)
            for page in reader.pages:
                content = page.extract_text()
                if content:
                    text += content.strip() + "\n"

            # Prompt for the model
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

            # Extract questions using regex
            questions = re.findall(r"(?:^|\n)\s*\d[\.\)]\s*(.+)", response)

            # fallback if regex returns nothing
            if not questions:
                questions = [q.strip() for q in response.split("\n") if q.strip()]

    return render_template_string(html_template, questions=questions)

if __name__ == "__main__":
    app.run(debug=True)
