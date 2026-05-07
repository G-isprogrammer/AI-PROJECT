import os
import re
import json
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CATEGORIES = {
    "structural": ["crack", "wall crack", "column", "beam", "foundation", "ceiling crack", "شق", "تشققات", "جدار", "عمود", "سقف", "أساس", "خرسانة"],
    "electrical": ["electric", "electricity", "power", "light", "socket", "switch", "wiring", "كهرباء", "تماس", "سلك", "اسلاك", "فيش", "مفتاح", "إنارة", "لمبة"],
    "mechanical": ["pipe", "leak", "water", "drain", "drainage", "hvac", "ac", "air conditioning", "سباكة", "مويه", "ماء", "تسريب", "مكيف", "تصريف", "صرف", "أنبوب"],
    "safety": ["danger", "unsafe", "hazard", "fire", "smoke", "collapse", "خطر", "غير آمن", "سلامة", "حريق", "دخان", "انهيار"],
    "usability": ["door", "window", "handle", "paint", "finish", "tile", "noise", "باب", "نافذة", "مقبض", "دهان", "تشطيب", "بلاط", "إزعاج", "سيراميك"],
    "environmental": ["mold", "humidity", "dust", "smell", "ventilation", "عفن", "رطوبة", "غبار", "ريحة", "تهوية"],
    "contractual": ["contract", "specification", "requirement", "agreed", "not included", "عقد", "مواصفات", "شرط", "متفق", "غير مشمول", "غير مطابق"]
}

CRITICAL_WORDS = ["danger", "unsafe", "fire", "collapse", "critical", "خطر", "غير آمن", "حريق", "انهيار", "حرج"]
MAJOR_WORDS = ["leak", "broken", "failure", "serious", "problem", "تسريب", "مكسور", "عطل", "مشكلة", "خلل", "تشقق", "تشققات"]


def clean_text(text):
    text = (text or "").strip().lower()
    return re.sub(r"\s+", " ", text)


def detect_category(text):
    scores = {}
    for category, keywords in CATEGORIES.items():
        scores[category] = sum(1 for word in keywords if word in text)

    best_category = max(scores, key=scores.get)
    if scores[best_category] == 0:
        return "other", scores

    return best_category, scores


def detect_priority(text):
    if any(word in text for word in CRITICAL_WORDS):
        return "Critical"
    if any(word in text for word in MAJOR_WORDS):
        return "Major"
    return "Minor"


def detect_recurrence(text):
    recurrence_words = [
        "again", "repeated", "many times", "still", "same issue",
        "مرة ثانية", "متكرر", "يتكرر", "لسه", "نفس المشكلة"
    ]
    return any(word in text for word in recurrence_words)


def basic_feedback_analysis(text):
    cleaned = clean_text(text)
    category, scores = detect_category(cleaned)
    priority = detect_priority(cleaned)
    recurring = detect_recurrence(cleaned)

    if recurring and priority == "Major":
        priority = "Critical"

    return {
        "original_text": text,
        "cleaned_text": cleaned,
        "category": category,
        "priority": priority,
        "recurring": recurring,
        "scores": scores,
        "summary": cleaned[:250] + ("..." if len(cleaned) > 250 else "")
    }


def parse_json_response(raw_text):
    raw_text = (raw_text or "").strip()
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start != -1 and end != -1:
            return json.loads(raw_text[start:end + 1])
    raise ValueError("Feedback AI did not return valid JSON.")


def analyze_feedback_with_contract(feedback_text, contract_analysis):
    basic = basic_feedback_analysis(feedback_text)

    prompt = f"""
You are a construction contract and building defect analyst.

You are given:
1) A user feedback/review about a building issue.
2) A contract analysis containing legal clauses, construction scope, materials, technical requirements, and risks.

Your job:
- Understand the issue in the feedback.
- Decide whether the issue is related to the contract scope or specifications.
- Match it to the correct construction category, material, clause, or technical requirement.
- Explain the reasoning clearly.
- Do NOT guess.
- If the contract analysis does not mention the issue, say it is not covered or unclear.

Return ONLY valid JSON:

{{
  "issue_summary": "",
  "category": "",
  "priority": "",
  "recurring": false,
  "is_related_to_contract": true,
  "relation_status": "covered or not_covered or unclear",
  "matched_contract_area": "",
  "matched_evidence": "",
  "reason": "",
  "recommended_action": "",
  "confidence": "Low or Medium or High"
}}

Basic feedback classification:
{basic}

Feedback:
{feedback_text}

Contract analysis:
{contract_analysis}
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You analyze construction feedback against contract scope."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    ai_result = parse_json_response(response.choices[0].message.content)

    ai_result["basic_analysis"] = basic
    return ai_result


# Backward compatible name
def analyze_feedback(text):
    return basic_feedback_analysis(text)