import os
import re
import json
import cv2
import fitz
import numpy as np
import pandas as pd
import pytesseract

from docx import Document
from pdf2image import convert_from_path
from google import genai


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\poppler\poppler-25.12.0\Library\bin"

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY was not found. Check your .env file.")

client = genai.Client(api_key=API_KEY)


def clean_text(text):
    text = text or ""
    return re.sub(r"\s+", " ", text).strip()


def contains_arabic(text):
    return bool(re.search(r"[\u0600-\u06FF]", text or ""))


def preprocess_image_for_ocr(pil_image):
    img = np.array(pil_image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img


def extract_text_from_pdf(file_path):
    extracted_text = ""

    try:
        doc = fitz.open(file_path)
        for page in doc:
            extracted_text += page.get_text("text") + "\n"
        doc.close()

        extracted_text = clean_text(extracted_text)

        # If Arabic text looks problematic, OCR may work better
        if len(extracted_text) > 80 and not contains_arabic(extracted_text):
            return {
                "text": extracted_text,
                "method": "pdf_text"
            }

    except Exception:
        pass

    try:
        images = convert_from_path(file_path, poppler_path=POPPLER_PATH)
        ocr_text = ""

        for img in images:
            processed = preprocess_image_for_ocr(img)
            config = r"--oem 3 --psm 6"
            page_text = pytesseract.image_to_string(
                processed,
                lang="ara+eng",
                config=config
            )
            ocr_text += page_text + "\n"

        ocr_text = clean_text(ocr_text)

        if len(ocr_text) > 20:
            return {
                "text": ocr_text,
                "method": "ocr"
            }

    except Exception:
        pass

    return {
        "text": extracted_text,
        "method": "pdf_text_fallback"
    }


def extract_text_from_docx(file_path):
    try:
        doc = Document(file_path)
        text = ""

        for para in doc.paragraphs:
            text += para.text + "\n"

        return {
            "text": clean_text(text),
            "method": "docx"
        }

    except Exception:
        return {
            "text": "",
            "method": "docx_failed"
        }


def extract_text_from_excel(file_path):
    try:
        df = pd.read_excel(file_path)
        text_parts = []

        for col in df.columns:
            values = df[col].astype(str).tolist()
            text_parts.append(f"{col}: " + " ".join(values))

        return {
            "text": clean_text(" ".join(text_parts)),
            "method": "excel"
        }

    except Exception:
        return {
            "text": "",
            "method": "excel_failed"
        }


def default_result():
    return {
        "contract_type": "Unknown",
        "summary": [],
        "parties": [],
        "dates": [],
        "financial_terms": [],
        "clauses": {
            "payment": {"status": "missing", "evidence": []},
            "termination": {"status": "missing", "evidence": []},
            "liability": {"status": "missing", "evidence": []},
            "confidentiality": {"status": "missing", "evidence": []},
            "governing_law": {"status": "missing", "evidence": []},
            "penalties": {"status": "missing", "evidence": []},
            "renewal": {"status": "missing", "evidence": []}
        },
        "risks": [],
        "overall_risk": "Unknown",
        "recommendations": [],
        "confidence": "Low"
    }


def parse_json_response(raw_text):
    raw_text = (raw_text or "").strip()

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        start = raw_text.find("{")
        end = raw_text.rfind("}")

        if start != -1 and end != -1 and end > start:
            return json.loads(raw_text[start:end + 1])

    raise ValueError("Gemini did not return valid JSON.")


def analyze_with_gemini(contract_text):
    contract_text = clean_text(contract_text)

    if not contract_text:
        result = default_result()
        result["recommendations"] = ["No readable text was extracted from the file."]
        return result

    prompt = f"""
You are an expert contract analysis assistant.

Analyze the contract text and return ONLY valid JSON.
Do not include markdown.
Do not include explanations outside JSON.

Support both Arabic and English.

Be conservative:
- If a clause is unclear, mark it as "missing".
- Evidence must be short and copied from the contract text.
- Do not invent names, dates, or amounts.
- overall_risk must be exactly: Low, Medium, or High.
- confidence must be exactly: Low, Medium, or High.

Return this exact JSON structure:

{{
  "contract_type": "string",
  "summary": ["string", "string", "string"],
  "parties": ["string"],
  "dates": ["string"],
  "financial_terms": ["string"],
  "clauses": {{
    "payment": {{"status": "found or missing", "evidence": ["string"]}},
    "termination": {{"status": "found or missing", "evidence": ["string"]}},
    "liability": {{"status": "found or missing", "evidence": ["string"]}},
    "confidentiality": {{"status": "found or missing", "evidence": ["string"]}},
    "governing_law": {{"status": "found or missing", "evidence": ["string"]}},
    "penalties": {{"status": "found or missing", "evidence": ["string"]}},
    "renewal": {{"status": "found or missing", "evidence": ["string"]}}
  }},
  "risks": [
    {{"name": "string", "level": "Low or Medium or High", "reason": "string"}}
  ],
  "overall_risk": "Low or Medium or High",
  "recommendations": ["string"],
  "confidence": "Low or Medium or High"
}}

Contract text:
{contract_text}
"""

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )

    data = parse_json_response(response.text)

    result = default_result()
    result.update(data)

    return result


def analyze_contract(file_path=None, text=None):
    if file_path:
        lower_path = file_path.lower()

        if lower_path.endswith(".pdf"):
            extracted = extract_text_from_pdf(file_path)
        elif lower_path.endswith(".docx"):
            extracted = extract_text_from_docx(file_path)
        elif lower_path.endswith(".xlsx"):
            extracted = extract_text_from_excel(file_path)
        else:
            extracted = {
                "text": "",
                "method": "unsupported_file"
            }

        analysis = analyze_with_gemini(extracted["text"])
        analysis["extraction_method"] = extracted["method"]
        return analysis

    if text:
        analysis = analyze_with_gemini(text)
        analysis["extraction_method"] = "direct_text"
        return analysis

    result = default_result()
    result["extraction_method"] = "none"
    result["recommendations"] = ["No contract input was provided."]
    return result