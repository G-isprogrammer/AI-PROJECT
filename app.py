from flask import Flask, render_template, request
import os
from ai import extract_text, analyze_contract
from transformers import pipeline

app = Flask(__name__)

# 📁 uploads folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# 🤖 Chat AI (بسيط)
chatbot = pipeline("text-generation", model="distilgpt2")

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    risk = None
    summary = None
    explanation = None
    chat_response = None

    if request.method == "POST":
        file = request.files.get("file")
        user_question = request.form.get("question")

        # 📄 تحليل العقد
        if file and file.filename:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            text = extract_text(filepath)
            result, risk, summary, explanation = analyze_contract(text)

        # 🤖 Chat
        if user_question:
            response = chatbot(user_question, max_length=100, num_return_sequences=1)
            chat_response = response[0]["generated_text"]

    return render_template(
        "index.html",
        result=result,
        risk=risk,
        summary=summary,
        explanation=explanation,
        chat_response=chat_response
    )

if __name__ == "__main__":
    app.run(debug=True)