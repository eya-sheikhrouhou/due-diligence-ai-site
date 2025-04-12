from flask import Flask, render_template, request
import os
from chat_utils import get_chatbot_response

# 👉 Indique à Flask que les templates sont à la racine du projet
app = Flask(__name__, template_folder=os.getcwd())

@app.route("/")
def home():
    return render_template("index.html")  # Flask va chercher dans la racine maintenant

@app.route("/ask", methods=["POST"])
def ask():
    pdf = request.files["pdf"]
    question = request.form["question"]

    answer = get_chatbot_response(question, pdf)
    return {"answer": answer}

if __name__ == "__main__":
    app.run(debug=True)
