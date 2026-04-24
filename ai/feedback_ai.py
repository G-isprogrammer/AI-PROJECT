import re

CATEGORIES = {
    "structural": [
        "crack", "wall crack", "column", "beam", "foundation", "ceiling crack",
        "شق", "تشققات", "جدار", "عمود", "سقف", "أساس", "خرسانة"
    ],
    "electrical": [
        "electric", "electricity", "power", "light", "socket", "switch", "wiring",
        "كهرباء", "تماس", "سلك", "اسلاك", "فيش", "مفتاح", "إنارة", "لمبة"
    ],
    "mechanical": [
        "pipe", "leak", "water", "drain", "drainage", "hvac", "ac", "air conditioning",
        "سباكة", "مويه", "ماء", "تسريب", "مكيف", "تصريف", "صرف", "أنبوب"
    ],
    "safety": [
        "danger", "unsafe", "hazard", "fire", "smoke", "collapse",
        "خطر", "غير آمن", "سلامة", "حريق", "دخان", "انهيار"
    ],
    "usability": [
        "door", "window", "handle", "paint", "finish", "tile", "noise",
        "باب", "نافذة", "مقبض", "دهان", "تشطيب", "بلاط", "إزعاج"
    ],
    "environmental": [
        "mold", "humidity", "dust", "smell", "ventilation",
        "عفن", "رطوبة", "غبار", "ريحة", "تهوية"
    ],
    "contractual": [
        "contract", "specification", "requirement", "agreed", "not included",
        "عقد", "مواصفات", "شرط", "متفق", "غير مشمول", "غير مطابق"
    ]
}

CRITICAL_WORDS = [
    "danger", "unsafe", "fire", "collapse", "critical",
    "خطر", "غير آمن", "حريق", "انهيار", "حرج"
]

MAJOR_WORDS = [
    "leak", "broken", "failure", "serious", "problem",
    "تسريب", "مكسور", "عطل", "مشكلة", "خلل"
]


def clean_text(text):
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def detect_category(text):
    scores = {}
    for category, keywords in CATEGORIES.items():
        score = sum(1 for word in keywords if word in text)
        scores[category] = score

    best_category = max(scores, key=scores.get)
    if scores[best_category] == 0:
        return "other", scores

    return best_category, scores


def detect_priority(text):
    critical_hits = sum(1 for word in CRITICAL_WORDS if word in text)
    major_hits = sum(1 for word in MAJOR_WORDS if word in text)

    if critical_hits > 0:
        return "Critical"
    if major_hits > 0:
        return "Major"
    return "Minor"


def detect_recurrence(text):
    recurrence_words = [
        "again", "repeated", "many times", "still", "same issue",
        "مرة ثانية", "متكرر", "يتكرر", "لسه", "نفس المشكلة"
    ]
    return any(word in text for word in recurrence_words)


def analyze_feedback(text):
    cleaned = clean_text(text)
    category, scores = detect_category(cleaned)
    priority = detect_priority(cleaned)
    recurring = detect_recurrence(cleaned)

    if recurring and priority == "Major":
        priority = "Critical"

    summary = cleaned[:250] + ("..." if len(cleaned) > 250 else "")

    return {
        "original_text": text,
        "cleaned_text": cleaned,
        "category": category,
        "priority": priority,
        "recurring": recurring,
        "scores": scores,
        "summary": summary
    }