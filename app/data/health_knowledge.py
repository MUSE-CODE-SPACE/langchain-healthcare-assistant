"""
Health Knowledge Base - Medical information for the healthcare assistant
Note: This is for educational purposes only. Always consult healthcare professionals.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class SymptomSeverity(Enum):
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    EMERGENCY = "emergency"


@dataclass
class Symptom:
    """Represents a health symptom."""
    id: str
    name: str
    description: str
    common_causes: List[str]
    when_to_see_doctor: List[str]
    home_remedies: List[str]
    severity_indicators: Dict[str, SymptomSeverity]


@dataclass
class Condition:
    """Represents a health condition."""
    id: str
    name: str
    description: str
    common_symptoms: List[str]
    risk_factors: List[str]
    prevention: List[str]
    treatment_overview: str
    when_to_seek_help: List[str]


@dataclass
class Medication:
    """General medication information."""
    id: str
    name: str
    common_uses: List[str]
    general_dosage_info: str
    common_side_effects: List[str]
    warnings: List[str]
    interactions: List[str]


@dataclass
class WellnessTip:
    """Health and wellness tips."""
    category: str
    title: str
    description: str
    tips: List[str]


# =============================================================================
# SYMPTOMS DATABASE
# =============================================================================

SYMPTOMS_DB: Dict[str, Symptom] = {
    "headache": Symptom(
        id="headache",
        name="Headache",
        description="Pain or discomfort in the head, scalp, or neck area.",
        common_causes=[
            "Tension and stress",
            "Dehydration",
            "Lack of sleep",
            "Eye strain",
            "Sinus congestion",
            "Caffeine withdrawal",
            "Hunger or skipping meals"
        ],
        when_to_see_doctor=[
            "Severe or sudden onset headache",
            "Headache with fever, stiff neck, or confusion",
            "Headache after head injury",
            "Headache that worsens over days",
            "Headache with vision changes",
            "Frequent headaches affecting daily life"
        ],
        home_remedies=[
            "Rest in a quiet, dark room",
            "Stay hydrated - drink water",
            "Apply cold or warm compress",
            "Practice relaxation techniques",
            "Gentle neck and shoulder stretches",
            "Over-the-counter pain relievers (follow package directions)"
        ],
        severity_indicators={
            "mild_occasional": SymptomSeverity.MILD,
            "moderate_affecting_activities": SymptomSeverity.MODERATE,
            "severe_worst_ever": SymptomSeverity.SEVERE,
            "with_neurological_symptoms": SymptomSeverity.EMERGENCY
        }
    ),
    "fatigue": Symptom(
        id="fatigue",
        name="Fatigue",
        description="A feeling of tiredness, exhaustion, or lack of energy.",
        common_causes=[
            "Poor sleep quality",
            "Stress and anxiety",
            "Poor nutrition",
            "Sedentary lifestyle",
            "Dehydration",
            "Anemia",
            "Thyroid issues",
            "Depression"
        ],
        when_to_see_doctor=[
            "Fatigue lasting more than 2 weeks",
            "Fatigue with unexplained weight loss",
            "Fatigue with shortness of breath",
            "Fatigue affecting daily activities",
            "Fatigue with fever or other symptoms"
        ],
        home_remedies=[
            "Establish regular sleep schedule",
            "Exercise regularly (even light walking)",
            "Eat balanced meals",
            "Stay hydrated",
            "Limit caffeine and alcohol",
            "Practice stress management",
            "Take short breaks during work"
        ],
        severity_indicators={
            "occasional_tiredness": SymptomSeverity.MILD,
            "daily_fatigue": SymptomSeverity.MODERATE,
            "unable_to_function": SymptomSeverity.SEVERE
        }
    ),
    "sore_throat": Symptom(
        id="sore_throat",
        name="Sore Throat",
        description="Pain, scratchiness, or irritation of the throat.",
        common_causes=[
            "Viral infections (common cold, flu)",
            "Bacterial infections (strep throat)",
            "Allergies",
            "Dry air",
            "Smoking or secondhand smoke",
            "Voice strain",
            "Acid reflux"
        ],
        when_to_see_doctor=[
            "Severe pain or difficulty swallowing",
            "Sore throat lasting more than a week",
            "High fever (over 101°F/38.3°C)",
            "Swollen lymph nodes",
            "White patches on tonsils",
            "Difficulty breathing",
            "Blood in saliva or phlegm"
        ],
        home_remedies=[
            "Gargle with warm salt water",
            "Stay hydrated with warm liquids",
            "Use throat lozenges",
            "Rest your voice",
            "Use a humidifier",
            "Honey in warm tea (adults only)",
            "Over-the-counter pain relievers"
        ],
        severity_indicators={
            "mild_discomfort": SymptomSeverity.MILD,
            "moderate_pain": SymptomSeverity.MODERATE,
            "severe_difficulty_swallowing": SymptomSeverity.SEVERE,
            "difficulty_breathing": SymptomSeverity.EMERGENCY
        }
    ),
    "back_pain": Symptom(
        id="back_pain",
        name="Back Pain",
        description="Pain affecting the lower, middle, or upper back.",
        common_causes=[
            "Poor posture",
            "Muscle strain",
            "Prolonged sitting",
            "Heavy lifting",
            "Lack of exercise",
            "Obesity",
            "Age-related changes"
        ],
        when_to_see_doctor=[
            "Pain lasting more than 6 weeks",
            "Pain radiating down legs",
            "Numbness or tingling",
            "Pain after injury",
            "Pain with fever",
            "Difficulty with bladder or bowel control",
            "Unexplained weight loss with pain"
        ],
        home_remedies=[
            "Apply ice for first 48 hours, then heat",
            "Stay active with gentle movement",
            "Practice good posture",
            "Sleep on a supportive mattress",
            "Stretch regularly",
            "Strengthen core muscles",
            "Over-the-counter pain relievers"
        ],
        severity_indicators={
            "occasional_discomfort": SymptomSeverity.MILD,
            "daily_pain": SymptomSeverity.MODERATE,
            "severe_limiting_movement": SymptomSeverity.SEVERE,
            "with_neurological_symptoms": SymptomSeverity.EMERGENCY
        }
    ),
    "stomach_ache": Symptom(
        id="stomach_ache",
        name="Stomach Ache",
        description="Pain or discomfort in the abdominal area.",
        common_causes=[
            "Indigestion",
            "Gas and bloating",
            "Food intolerance",
            "Stress and anxiety",
            "Overeating",
            "Constipation",
            "Stomach flu (gastroenteritis)"
        ],
        when_to_see_doctor=[
            "Severe or persistent pain",
            "Pain with bloody stool",
            "Vomiting blood",
            "High fever with abdominal pain",
            "Pain during pregnancy",
            "Unable to keep food down",
            "Yellowing of skin or eyes"
        ],
        home_remedies=[
            "Rest and avoid solid food temporarily",
            "Stay hydrated with clear fluids",
            "Eat bland foods (BRAT diet)",
            "Avoid trigger foods",
            "Try ginger or peppermint tea",
            "Apply warm compress",
            "Avoid lying flat after eating"
        ],
        severity_indicators={
            "mild_discomfort": SymptomSeverity.MILD,
            "persistent_pain": SymptomSeverity.MODERATE,
            "severe_sharp_pain": SymptomSeverity.SEVERE,
            "with_danger_signs": SymptomSeverity.EMERGENCY
        }
    ),
    "cough": Symptom(
        id="cough",
        name="Cough",
        description="A sudden, forceful expulsion of air from the lungs.",
        common_causes=[
            "Common cold or flu",
            "Allergies",
            "Asthma",
            "Post-nasal drip",
            "Acid reflux",
            "Smoking",
            "Environmental irritants"
        ],
        when_to_see_doctor=[
            "Cough lasting more than 3 weeks",
            "Coughing up blood",
            "Shortness of breath",
            "Chest pain with coughing",
            "High fever",
            "Night sweats with cough",
            "Unexplained weight loss"
        ],
        home_remedies=[
            "Stay hydrated",
            "Use honey in warm tea (adults)",
            "Use a humidifier",
            "Avoid irritants (smoke, dust)",
            "Elevate head while sleeping",
            "Try over-the-counter cough medicine",
            "Gargle with salt water"
        ],
        severity_indicators={
            "occasional_cough": SymptomSeverity.MILD,
            "persistent_cough": SymptomSeverity.MODERATE,
            "severe_with_other_symptoms": SymptomSeverity.SEVERE,
            "difficulty_breathing": SymptomSeverity.EMERGENCY
        }
    )
}

# =============================================================================
# CONDITIONS DATABASE
# =============================================================================

CONDITIONS_DB: Dict[str, Condition] = {
    "common_cold": Condition(
        id="common_cold",
        name="Common Cold",
        description="A viral infection of the upper respiratory tract.",
        common_symptoms=["Runny nose", "Sneezing", "Sore throat", "Cough", "Mild body aches", "Low-grade fever"],
        risk_factors=["Exposure to infected individuals", "Weakened immune system", "Cold weather", "Stress", "Lack of sleep"],
        prevention=["Wash hands frequently", "Avoid touching face", "Stay away from sick people", "Get adequate sleep", "Maintain healthy diet"],
        treatment_overview="Rest, fluids, and symptom management. Most colds resolve in 7-10 days.",
        when_to_seek_help=["Symptoms lasting more than 10 days", "High fever", "Severe symptoms", "Difficulty breathing"]
    ),
    "flu": Condition(
        id="flu",
        name="Influenza (Flu)",
        description="A contagious respiratory illness caused by influenza viruses.",
        common_symptoms=["High fever", "Severe body aches", "Fatigue", "Cough", "Headache", "Chills"],
        risk_factors=["Not vaccinated", "Age (very young or elderly)", "Chronic conditions", "Weakened immune system"],
        prevention=["Annual flu vaccination", "Hand hygiene", "Avoid close contact with sick people", "Cover coughs and sneezes"],
        treatment_overview="Rest, fluids, and antiviral medications if prescribed early. Recovery typically takes 1-2 weeks.",
        when_to_seek_help=["Difficulty breathing", "Chest pain", "Confusion", "Severe vomiting", "Symptoms that improve then worsen"]
    ),
    "allergies": Condition(
        id="allergies",
        name="Allergies",
        description="Immune system reactions to substances that are usually harmless.",
        common_symptoms=["Sneezing", "Runny nose", "Itchy eyes", "Skin rash", "Congestion", "Watery eyes"],
        risk_factors=["Family history", "Having other allergies", "Asthma", "Environmental exposure"],
        prevention=["Identify and avoid triggers", "Keep windows closed during high pollen", "Use air purifiers", "Shower after outdoor activities"],
        treatment_overview="Avoid triggers, use antihistamines, nasal sprays, or immunotherapy as recommended.",
        when_to_seek_help=["Severe reactions (anaphylaxis)", "Difficulty breathing", "Swelling of face or throat", "Symptoms not controlled by OTC medications"]
    ),
    "hypertension": Condition(
        id="hypertension",
        name="High Blood Pressure (Hypertension)",
        description="A condition where the force of blood against artery walls is too high.",
        common_symptoms=["Often no symptoms", "Headaches (in severe cases)", "Shortness of breath", "Nosebleeds"],
        risk_factors=["Age", "Family history", "Obesity", "Sedentary lifestyle", "High sodium diet", "Excessive alcohol", "Stress"],
        prevention=["Maintain healthy weight", "Exercise regularly", "Eat low-sodium diet", "Limit alcohol", "Manage stress", "Don't smoke"],
        treatment_overview="Lifestyle modifications and medications as prescribed. Regular monitoring is important.",
        when_to_seek_help=["Blood pressure over 180/120", "Severe headache", "Chest pain", "Vision problems", "Difficulty breathing"]
    ),
    "diabetes_type2": Condition(
        id="diabetes_type2",
        name="Type 2 Diabetes",
        description="A chronic condition affecting how the body processes blood sugar.",
        common_symptoms=["Increased thirst", "Frequent urination", "Fatigue", "Blurred vision", "Slow-healing wounds", "Unexplained weight loss"],
        risk_factors=["Obesity", "Sedentary lifestyle", "Family history", "Age over 45", "History of gestational diabetes"],
        prevention=["Maintain healthy weight", "Regular exercise", "Balanced diet", "Limit processed foods", "Regular health screenings"],
        treatment_overview="Lifestyle changes, blood sugar monitoring, medications, and insulin if needed.",
        when_to_seek_help=["Very high or low blood sugar", "Confusion", "Difficulty breathing", "Fruity breath odor", "Unconsciousness"]
    )
}

# =============================================================================
# WELLNESS TIPS DATABASE
# =============================================================================

WELLNESS_TIPS: List[WellnessTip] = [
    WellnessTip(
        category="nutrition",
        title="Healthy Eating Basics",
        description="Fundamental principles for a balanced diet.",
        tips=[
            "Eat a variety of colorful fruits and vegetables daily",
            "Choose whole grains over refined grains",
            "Include lean proteins in your meals",
            "Stay hydrated - aim for 8 glasses of water daily",
            "Limit processed foods and added sugars",
            "Practice portion control",
            "Don't skip breakfast",
            "Plan and prepare meals ahead of time"
        ]
    ),
    WellnessTip(
        category="exercise",
        title="Physical Activity Guidelines",
        description="Recommendations for staying active.",
        tips=[
            "Aim for 150 minutes of moderate exercise per week",
            "Include strength training 2+ times per week",
            "Take breaks from sitting every 30-60 minutes",
            "Find activities you enjoy",
            "Start slowly and gradually increase intensity",
            "Include stretching and flexibility exercises",
            "Walk more - take stairs, park farther away",
            "Consider a workout buddy for motivation"
        ]
    ),
    WellnessTip(
        category="sleep",
        title="Better Sleep Habits",
        description="Tips for improving sleep quality.",
        tips=[
            "Maintain consistent sleep and wake times",
            "Create a relaxing bedtime routine",
            "Keep bedroom cool, dark, and quiet",
            "Avoid screens 1 hour before bed",
            "Limit caffeine after noon",
            "Avoid large meals close to bedtime",
            "Exercise regularly but not too close to bedtime",
            "Manage stress with relaxation techniques"
        ]
    ),
    WellnessTip(
        category="mental_health",
        title="Mental Wellness",
        description="Strategies for mental and emotional well-being.",
        tips=[
            "Practice mindfulness or meditation",
            "Stay connected with friends and family",
            "Set realistic goals and expectations",
            "Take breaks and time for yourself",
            "Limit news and social media consumption",
            "Seek help when needed - it's a sign of strength",
            "Practice gratitude daily",
            "Engage in hobbies and activities you enjoy"
        ]
    ),
    WellnessTip(
        category="preventive_care",
        title="Preventive Health Care",
        description="Important preventive health measures.",
        tips=[
            "Schedule regular check-ups with your doctor",
            "Keep vaccinations up to date",
            "Know your family health history",
            "Get recommended health screenings",
            "Monitor your blood pressure",
            "Maintain a healthy weight",
            "Don't smoke and limit alcohol",
            "Practice safe sun exposure"
        ]
    )
]


def get_symptom(symptom_id: str) -> Optional[Symptom]:
    """Get symptom by ID."""
    return SYMPTOMS_DB.get(symptom_id)


def get_condition(condition_id: str) -> Optional[Condition]:
    """Get condition by ID."""
    return CONDITIONS_DB.get(condition_id)


def search_symptoms(query: str) -> List[Symptom]:
    """Search symptoms by keyword."""
    query_lower = query.lower()
    results = []
    for symptom in SYMPTOMS_DB.values():
        if (query_lower in symptom.name.lower() or
            query_lower in symptom.description.lower() or
            any(query_lower in cause.lower() for cause in symptom.common_causes)):
            results.append(symptom)
    return results


def search_conditions(query: str) -> List[Condition]:
    """Search conditions by keyword."""
    query_lower = query.lower()
    results = []
    for condition in CONDITIONS_DB.values():
        if (query_lower in condition.name.lower() or
            query_lower in condition.description.lower() or
            any(query_lower in symptom.lower() for symptom in condition.common_symptoms)):
            results.append(condition)
    return results


def get_wellness_tips_by_category(category: str) -> List[WellnessTip]:
    """Get wellness tips by category."""
    return [tip for tip in WELLNESS_TIPS if tip.category == category]
