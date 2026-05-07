import os
<<<<<<< HEAD
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from ai.contract_ai import analyze_contract
=======
from flask import Flask, render_template, request, session
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

load_dotenv()
>>>>>>> 2bf1007 (Improve contract clause analysis and recommendations)

from ai.contract_ai import analyze_contract
from ai.feedback_ai import analyze_feedback_with_contract

app = Flask(__name__)

app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf", "docx", "xlsx", "png", "jpg", "jpeg"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


@app.route("/")
def home():
    return render_template("contract.html")


@app.route("/contract", methods=["GET", "POST"])
def contract_page():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template(
                "contract.html",
                error="No file was uploaded."
            )

        file = request.files["file"]

        if file.filename == "":
            return render_template(
                "contract.html",
                error="Please choose a file."
            )

        if not allowed_file(file.filename):
            return render_template(
                "contract.html",
                error="Only PDF, DOCX, XLSX, PNG, JPG, and JPEG files are allowed."
            )

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

        try:
            file.save(file_path)

            analysis = analyze_contract(file_path=file_path)

            session["last_contract_analysis"] = analysis

            return render_template(
                "contract.html",
                analysis=analysis,
                filename=filename
            )

        except Exception as e:
            return render_template(
                "contract.html",
                error=f"Analysis failed: {str(e)}"
            )

        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

    return render_template("contract.html")


@app.route("/feedback", methods=["GET", "POST"])
def feedback_page():
    if request.method == "POST":
        feedback_text = request.form.get("feedback", "").strip()

        if not feedback_text:
            return render_template(
                "feedback.html",
                error="Please enter feedback."
            )

        try:
            contract_analysis = session.get(
                "last_contract_analysis",
                "No contract analysis available yet."
            )

            result = analyze_feedback_with_contract(
                feedback_text,
                contract_analysis
            )

            return render_template(
                "feedback.html",
                result=result,
                feedback_text=feedback_text
            )

        except Exception as e:
            return render_template(
                "feedback.html",
                error=f"Feedback analysis failed: {str(e)}",
                feedback_text=feedback_text
            )

    return render_template("feedback.html")


if __name__ == "__main__":
    app.run(debug=True)