FIDIC_KNOWLEDGE = """
Analyze the contract using FIDIC-inspired construction contract principles.

A strong construction contract should include:
1. Clear parties: Employer/Owner, Contractor, and Engineer/Consultant if applicable.
2. Clear contract documents: Contract Agreement, Tender, Specifications, Drawings, Schedules, BOQ.
3. Clear scope of work: project description, location, drawings, materials, and technical requirements.
4. Clear contract price and payment terms: total value, currency, payment schedule, advance payment, interim payment, retention money, final payment.
5. Clear time for completion: commencement date, project duration, completion date, and extension of time rules.
6. Delay damages or penalties for late completion.
7. Contractor responsibilities: execution, materials, labor, equipment, quality, safety, and remedying defects.
8. Employer responsibilities: site access, approvals when applicable, and payment obligations.
9. Variations/change orders must be written and should explain impact on time and cost.
10. Defects liability or warranty period.
11. Insurance requirements.
12. Performance security or performance bond.
13. Force majeure clause.
14. Dispute resolution clause.
15. Compliance with applicable laws, permits, regulations, and Saudi Building Code when the project is in Saudi Arabia.
"""

SCORING_RULES = {
    "parties": {
        "weight": 10,
        "keywords": [
            "owner", "employer", "contractor", "consultant", "engineer",
            "المالك", "صاحب العمل", "المقاول", "الاستشاري", "المهندس"
        ],
        "description": "Clear identification of contract parties"
    },
    "contract_documents": {
        "weight": 6,
        "keywords": [
            "contract agreement", "specification", "drawings", "bill of quantities", "boq", "tender",
            "اتفاقية العقد", "المواصفات", "الرسومات", "جداول الكميات", "العطاء"
        ],
        "description": "Contract documents are mentioned"
    },
    "scope_of_work": {
        "weight": 10,
        "keywords": [
            "scope of work", "works", "construction", "execute", "project", "materials",
            "نطاق العمل", "الأعمال", "تنفيذ", "مشروع", "مواد", "إنشاء", "بناء"
        ],
        "description": "Scope of work is clear"
    },
    "payment_terms": {
        "weight": 12,
        "keywords": [
            "contract price", "payment", "advance payment", "interim payment", "final payment",
            "retention", "invoice", "amount", "sar", "riyal",
            "قيمة العقد", "الدفع", "دفعة", "المستخلص", "الدفعة النهائية", "احتجاز", "ريال"
        ],
        "description": "Payment terms are clear"
    },
    "time_for_completion": {
        "weight": 10,
        "keywords": [
            "commencement date", "completion date", "time for completion", "duration", "days", "months",
            "تاريخ البدء", "تاريخ الانتهاء", "مدة المشروع", "مدة التنفيذ", "أيام", "شهور", "أشهر"
        ],
        "description": "Time for completion is defined"
    },
    "delay_penalties": {
        "weight": 8,
        "keywords": [
            "delay damages", "liquidated damages", "penalty", "late completion",
            "غرامة", "غرامات التأخير", "تأخير", "جزاء", "تعويض التأخير"
        ],
        "description": "Delay penalties are included"
    },
    "contractor_responsibilities": {
        "weight": 8,
        "keywords": [
            "contractor shall", "contractor is responsible", "labor", "equipment", "quality", "safety",
            "يلتزم المقاول", "مسؤولية المقاول", "العمالة", "المعدات", "الجودة", "السلامة"
        ],
        "description": "Contractor responsibilities are defined"
    },
    "employer_responsibilities": {
        "weight": 6,
        "keywords": [
            "employer shall", "owner shall", "site access", "approvals", "permits",
            "يلتزم المالك", "مسؤولية المالك", "تسليم الموقع", "التصاريح", "الموافقات"
        ],
        "description": "Employer responsibilities are defined"
    },
    "variations": {
        "weight": 7,
        "keywords": [
            "variation", "change order", "change in scope", "written approval",
            "تعديل", "أمر تغيير", "تغيير نطاق العمل", "موافقة خطية"
        ],
        "description": "Variation/change order clause exists"
    },
    "defects_liability": {
        "weight": 7,
        "keywords": [
            "defects liability", "warranty", "defect", "remedying defects", "maintenance period",
            "ضمان", "العيوب", "إصلاح العيوب", "فترة الصيانة"
        ],
        "description": "Defects liability or warranty clause exists"
    },
    "insurance": {
        "weight": 5,
        "keywords": [
            "insurance", "third party liability", "workers compensation",
            "تأمين", "المسؤولية تجاه الغير", "إصابات العمال"
        ],
        "description": "Insurance clause exists"
    },
    "performance_security": {
        "weight": 5,
        "keywords": [
            "performance security", "performance bond", "bank guarantee",
            "ضمان الأداء", "ضمان بنكي", "خطاب ضمان"
        ],
        "description": "Performance security clause exists"
    },
    "force_majeure": {
        "weight": 4,
        "keywords": [
            "force majeure", "exceptional event", "beyond control",
            "القوة القاهرة", "ظروف خارجة عن السيطرة", "حدث استثنائي"
        ],
        "description": "Force majeure clause exists"
    },
    "dispute_resolution": {
        "weight": 5,
        "keywords": [
            "dispute", "arbitration", "court", "mediation", "claim",
            "نزاع", "تحكيم", "محكمة", "مطالبة", "تسوية"
        ],
        "description": "Dispute resolution clause exists"
    },
    "saudi_compliance": {
        "weight": 2,
        "keywords": [
            "saudi", "saudi building code", "municipality", "permit", "sbc",
            "السعودية", "كود البناء السعودي", "البلدية", "رخصة", "تصريح"
        ],
        "description": "Saudi compliance is mentioned"
    }
}