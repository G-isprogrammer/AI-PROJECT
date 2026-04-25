import os
import re
import json
import fitz
import pandas as pd
import pytesseract
import cv2
import numpy as np

from docx import Document
from pdf2image import convert_from_path
from google import genai

# 🔑 Gemini
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ⚠️ عدلي المسار إذا عندك مختلف
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = None

# -------------------------------
# تنظيف النص
# -------------------------------
def clean_text(text):
    return re.sub(r"\s+", " ", text or "").strip()


# -------------------------------
# PDF Extraction (ذكي)
# -------------------------------
def extract_text_from_pdf(file_path):
    text = ""

    try:
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text()
        doc.close()

        text = clean_text(text)

        # لو النص فاضي → OCR
        if len(text) < 50:
            images = convert_from_path(file_path, poppler_path=POPPLER_PATH)

            for img in images:
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
                text += pytesseract.image_to_string(img, lang="ara+eng")

        return text

    except Exception as e:
        print("PDF Error:", e)
        return ""


# -------------------------------
# Excel
# -------------------------------
def extract_text_from_excel(file_path):
    try:
        df = pd.read_excel(file_path)
        return clean_text(" ".join(df.astype(str).values.flatten()))
    except:
        return ""


# -------------------------------
# Word
# -------------------------------
def extract_text_from_docx(file_path):
    try:
        doc = Document(file_path)
        return clean_text(" ".join([p.text for p in doc.paragraphs]))
    except:
        return ""


# -------------------------------
# Gemini LLM
# -------------------------------
def default_result():
    return {
        "contract_type": "Unknown",
        "summary": [],
        "parties": [],
        "dates": [],
        "financial_terms": [],
        "clauses": {
            "payment": {"status": "missing", "score": "", "matched_by": "LLM", "evidence": []},
            "termination": {"status": "missing", "score": "", "matched_by": "LLM", "evidence": []},
            "liability": {"status": "missing", "score": "", "matched_by": "LLM", "evidence": []},
            "confidentiality": {"status": "missing", "score": "", "matched_by": "LLM", "evidence": []},
            "governing_law": {"status": "missing", "score": "", "matched_by": "LLM", "evidence": []},
            "penalties": {"status": "missing", "score": "", "matched_by": "LLM", "evidence": []},
            "renewal": {"status": "missing", "score": "", "matched_by": "LLM", "evidence": []}
        },
        "risks": [],
        "overall_risk": "Unknown",
        "risk_score": "",
        "recommendations": [],
        "confidence": "Low",
        "extraction_method": ""
    }


def normalize_result(data):
    result = default_result()

    if not isinstance(data, dict):
        result["recommendations"] = ["AI returned an invalid response."]
        return result

    result["contract_type"] = data.get("contract_type", "Unknown")
    result["summary"] = data.get("summary", [])
    result["parties"] = data.get("parties", [])
    result["dates"] = data.get("dates", [])
    result["financial_terms"] = data.get("financial_terms", [])
    result["risks"] = data.get("risks", [])
    result["overall_risk"] = data.get("overall_risk", "Unknown")
    result["risk_score"] = data.get("risk_score", "")
    result["recommendations"] = data.get("recommendations", [])
    result["confidence"] = data.get("confidence", "Low")

    clauses = data.get("clauses", {})
    for name in result["clauses"]:
        clause = clauses.get(name, {})
        result["clauses"][name] = {
            "status": clause.get("status", "missing"),
            "score": clause.get("score", ""),
            "matched_by": clause.get("matched_by", "LLM"),
            "evidence": clause.get("evidence", [])
        }

    return result


def analyze_with_gemini(text):
    if not text or len(text.strip()) < 20:
        result = default_result()
        result["recommendations"] = ["No readable text was extracted from the uploaded file."]
        return result

    prompt = f"""
You are an expert contract analysis assistant.

Return ONLY valid JSON. No markdown. No explanation.

Analyze this contract and return this exact structure:

{{
  "contract_type": "string",
  "summary": ["string", "string", "string"],
  "parties": ["string"],
  "dates": ["string"],
  "financial_terms": ["string"],
  "clauses": {{
    "payment": {{"status": "found or missing", "score": "High/Medium/Low", "matched_by": "LLM", "evidence": ["string"]}},
    "termination": {{"status": "found or missing", "score": "High/Medium/Low", "matched_by": "LLM", "evidence": ["string"]}},
    "liability": {{"status": "found or missing", "score": "High/Medium/Low", "matched_by": "LLM", "evidence": ["string"]}},
    "confidentiality": {{"status": "found or missing", "score": "High/Medium/Low", "matched_by": "LLM", "evidence": ["string"]}},
    "governing_law": {{"status": "found or missing", "score": "High/Medium/Low", "matched_by": "LLM", "evidence": ["string"]}},
    "penalties": {{"status": "found or missing", "score": "High/Medium/Low", "matched_by": "LLM", "evidence": ["string"]}},
    "renewal": {{"status": "found or missing", "score": "High/Medium/Low", "matched_by": "LLM", "evidence": ["string"]}}
  }},
  "risks": [
    {{"name": "string", "level": "Low/Medium/High", "reason": "string"}}
  ],
  "overall_risk": "Low/Medium/High",
  "risk_score": "number from 0 to 10",
  "recommendations": ["string"],
  "confidence": "Low/Medium/High"
}}

Contract text:
{text}
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )

        raw = (response.text or "").strip()

        start = raw.find("{")
        end = raw.rfind("}")

        if start == -1 or end == -1:
            result = default_result()
            result["recommendations"] = ["Gemini did not return JSON."]
            return result

        data = json.loads(raw[start:end + 1])
        return normalize_result(data)

    except Exception as e:
        result = default_result()
        result["recommendations"] = [f"Gemini analysis failed: {str(e)}"]
        return result


def analyze_contract(file_path=None):
    if not file_path:
        result = default_result()
        result["recommendations"] = ["No file was uploaded."]
        return result

    lower_path = file_path.lower()

    if lower_path.endswith(".pdf"):
        text = extract_text_from_pdf(file_path)
        method = "pdf"

    elif lower_path.endswith(".docx"):
        text = extract_text_from_docx(file_path)
        method = "docx"

    elif lower_path.endswith(".xlsx"):
        text = extract_text_from_excel(file_path)
        method = "excel"

    else:
        result = default_result()
        result["extraction_method"] = "unsupported_file"
        result["recommendations"] = ["Unsupported file type."]
        return result

    analysis = analyze_with_gemini(text)
    analysis["extraction_method"] = method

    print("TEXT LENGTH:", len(text))
    print("EXTRACTION METHOD:", method)
    print("AI RESULT KEYS:", analysis.keys())

    return analysis