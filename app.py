import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from ai.contract_ai import analyze_contract

load_dotenv()

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf", "docx", "xlsx"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def home():
    return render_template("contract.html")


@app.route("/contract", methods=["GET", "POST"])
def contract_page():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("contract.html", error="No file was uploaded.")

        file = request.files["file"]

        if file.filename == "":
            return render_template("contract.html", error="Please choose a file.")

        if not allowed_file(file.filename):
            return render_template(
                "contract.html",
                error="Only PDF, DOCX, and XLSX files are allowed."
            )

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        try:
            analysis = analyze_contract(file_path=file_path)

            # Security: delete uploaded contract after analysis
            if os.path.exists(file_path):
                os.remove(file_path)

            return render_template(
                "contract.html",
                analysis=analysis,
                filename=filename
            )

        except Exception as e:
            if os.path.exists(file_path):
                os.remove(file_path)

            return render_template(
                "contract.html",
                error=f"Analysis failed: {str(e)}"
            )

    return render_template("contract.html")


if __name__ == "__main__":
    app.run(debug=True)