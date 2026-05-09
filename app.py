import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

load_dotenv()

from ai.contract_ai import analyze_contract
from ai.feedback_ai import analyze_feedback_with_contract

app = Flask(__name__)
CORS(app)

app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf", "docx", "xlsx", "png", "jpg", "jpeg"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/api/analyze-contract", methods=["POST"])
def analyze_contract_api():
    if "file" not in request.files:
        return jsonify({"error": "No file was uploaded."}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Please choose a file."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Only PDF, DOCX, XLSX, PNG, JPG, and JPEG files are allowed."}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    try:
        file.save(file_path)
        analysis = analyze_contract(file_path=file_path)
        return jsonify({"analysis": analysis}), 200

    except Exception as e:
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@app.route("/api/analyze-feedback", methods=["POST"])
def analyze_feedback_api():
    data = request.get_json() or {}
    feedback_text = data.get("feedback", "").strip()
    contract_analysis = data.get("contract_analysis", "")

    if not feedback_text:
        return jsonify({"error": "Please enter feedback."}), 400

    try:
        result = analyze_feedback_with_contract(feedback_text, contract_analysis)
        return jsonify({"result": result}), 200

    except Exception as e:
        return jsonify({"error": f"Feedback analysis failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)