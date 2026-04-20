import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from sentence_transformers import SentenceTransformer, util


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

POPPLER_PATH = r"C:\poppler\poppler-25.12.0\Library\bin"


model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v1")

CLAUSE_EXAMPLES = {
    "payment": [
        "payment terms", "payment schedule", "contract value", "installments",
        "stage payment", "final payment", "advance payment",
        "قيمة العقد", "الدفعات", "مراحل الدفع", "دفعة", "يتم سداد", "مستحقات"
    ],
    "termination": [
        "termination of contract", "end of contract", "cancel agreement",
        "contract may be terminated", "dispute resolution",
        "فسخ العقد", "إنهاء العقد", "إلغاء العقد", "في حال الإخلال", "فض النزاعات"
    ],
    "liability": [
        "liability for damages", "responsibility for damages",
        "contractor responsibilities", "owner responsibilities",
        "obligations of the contractor", "obligations of the owner",
        "المسؤولية", "الأضرار", "يتحمل المسؤولية",
        "التزامات المقاول", "التزامات المالك"
    ]
}

KEYWORDS = {
    "payment": [
        "payment", "installment", "contract value", "final payment",
        "دفعة", "الدفعات", "قيمة العقد", "مستحقات", "يتم سداد"
    ],
    "termination": [
        "termination", "cancel", "end of contract",
        "فسخ", "إنهاء العقد", "إلغاء العقد"
    ],
    "liability": [
        "liability", "responsibility", "responsibilities", "obligations", "damages",
        "مسؤولية", "المسؤولية", "الأضرار", "التزامات المقاول", "التزامات المالك"
    ]
}

def extract_text(file_path):
    try:
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += (page.extract_text() or "") + "\n"

        if len(text.strip()) > 50:
            return text
    except Exception:
        pass

    images = convert_from_path(file_path, poppler_path=POPPLER_PATH)
    text = ""

    for img in images:
        text += pytesseract.image_to_string(img, lang="ara+eng") + "\n"

    return text

def split_text(text, chunk_size=40):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i + chunk_size]))

    return chunks if chunks else [""]

def summarize(text):
    clean = " ".join(text.split())
    return clean[:300] + ("..." if len(clean) > 300 else "")

def analyze_contract(text):
    chunks = split_text(text)
    chunk_embeddings = model.encode(chunks, convert_to_tensor=True)

    result = {}
    lower_text = text.lower()

    for clause_name, examples in CLAUSE_EXAMPLES.items():
        example_embeddings = model.encode(examples, convert_to_tensor=True)
        scores = util.cos_sim(example_embeddings, chunk_embeddings)
        best_score = float(scores.max().item())

        keyword_found = any(word.lower() in lower_text for word in KEYWORDS[clause_name])
        semantic_found = best_score >= 0.30
        found = semantic_found or keyword_found

        result[clause_name] = {
            "status": "found" if found else "missing",
            "score": round(best_score, 3)
        }

    missing = sum(1 for x in result.values() if x["status"] == "missing")

    if missing >= 2:
        risk = "High"
        explanation = "Risk is high because important clauses are missing."
    elif missing == 1:
        risk = "Medium"
        explanation = "Risk is medium because one clause is missing."
    else:
        risk = "Low"
        explanation = "Risk is low because all main clauses are present."

    return result, risk, summarize(text), explanation