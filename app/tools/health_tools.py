"""
Health Tools - LangChain tools for the healthcare assistant.

DISCLAIMER: For informational purposes only. NOT medical advice.
"""

from __future__ import annotations

import json

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from app.data.health_knowledge import (
    CONDITIONS_DB,
    SYMPTOMS_DB,
    WELLNESS_TIPS,
    get_wellness_tips_by_category,
    search_conditions,
    search_symptoms,
)

DISCLAIMER = (
    "\n**IMPORTANT DISCLAIMER**: This information is for educational purposes only "
    "and is not a substitute for professional medical advice, diagnosis, or treatment. "
    "Always seek the advice of your physician or other qualified health provider with "
    "any questions you may have regarding a medical condition. Never disregard "
    "professional medical advice or delay in seeking it because of something you have "
    "read here. If you think you may have a medical emergency, call your doctor or "
    "emergency services immediately.\n"
)


class SymptomCheckerInput(BaseModel):
    """Input for symptom checker."""

    symptoms: list[str] = Field(description="List of symptoms to check")


class SymptomCheckerTool(BaseTool):
    """Tool for checking symptoms and getting general information."""

    name: str = "symptom_checker"
    description: str = (
        "Check symptoms and get general health information. NOT medical advice."
    )
    args_schema: type[BaseModel] = SymptomCheckerInput

    def _run(self, symptoms: list[str]) -> str:
        results = []

        for symptom_name in symptoms:
            symptom_key = symptom_name.lower().replace(" ", "_")
            symptom = SYMPTOMS_DB.get(symptom_key)

            if not symptom:
                found = search_symptoms(symptom_name)
                if found:
                    symptom = found[0]

            if symptom:
                results.append(
                    {
                        "symptom": symptom.name,
                        "description": symptom.description,
                        "common_causes": symptom.common_causes[:5],
                        "home_care": symptom.home_remedies[:4],
                        "when_to_see_doctor": symptom.when_to_see_doctor[:4],
                    }
                )

        if not results:
            return json.dumps(
                {
                    "message": "No matching symptoms found in our database.",
                    "recommendation": "Please consult a healthcare provider for personalized advice.",
                    "disclaimer": DISCLAIMER,
                },
                indent=2,
            )

        return json.dumps(
            {
                "symptoms_info": results,
                "general_advice": (
                    "Monitor your symptoms. If they worsen or persist, consult a healthcare provider."
                ),
                "disclaimer": DISCLAIMER,
            },
            indent=2,
        )


class ConditionInfoInput(BaseModel):
    """Input for condition information."""

    condition: str = Field(description="Name of the condition to look up")


class ConditionInfoTool(BaseTool):
    """Tool for getting information about health conditions."""

    name: str = "condition_info"
    description: str = "Get general information about health conditions. NOT medical advice."
    args_schema: type[BaseModel] = ConditionInfoInput

    def _run(self, condition: str) -> str:
        condition_key = condition.lower().replace(" ", "_")
        cond = CONDITIONS_DB.get(condition_key)

        if not cond:
            found = search_conditions(condition)
            if found:
                cond = found[0]

        if not cond:
            return json.dumps(
                {
                    "message": f"No information found for '{condition}'.",
                    "suggestion": "Try a different search term or consult a healthcare provider.",
                    "disclaimer": DISCLAIMER,
                },
                indent=2,
            )

        return json.dumps(
            {
                "condition": cond.name,
                "description": cond.description,
                "common_symptoms": cond.common_symptoms,
                "risk_factors": cond.risk_factors,
                "prevention": cond.prevention,
                "treatment_overview": cond.treatment_overview,
                "when_to_seek_help": cond.when_to_seek_help,
                "disclaimer": DISCLAIMER,
            },
            indent=2,
        )


class WellnessTipsInput(BaseModel):
    """Input for wellness tips."""

    category: str = Field(
        default="general",
        description="Category: nutrition, exercise, sleep, mental_health, preventive_care",
    )


class WellnessTipsTool(BaseTool):
    """Tool for getting health and wellness tips."""

    name: str = "wellness_tips"
    description: str = "Get health and wellness tips for various categories"
    args_schema: type[BaseModel] = WellnessTipsInput

    def _run(self, category: str = "general") -> str:
        if category == "general":
            all_tips = [
                {"category": tip.category, "title": tip.title, "tips": tip.tips[:3]}
                for tip in WELLNESS_TIPS
            ]
            return json.dumps(
                {
                    "wellness_tips": all_tips,
                    "message": "Here are general wellness tips for a healthier lifestyle.",
                },
                indent=2,
            )

        tips = get_wellness_tips_by_category(category)
        if not tips:
            return json.dumps(
                {
                    "message": f"No tips found for category '{category}'.",
                    "available_categories": [
                        "nutrition",
                        "exercise",
                        "sleep",
                        "mental_health",
                        "preventive_care",
                    ],
                },
                indent=2,
            )

        result = [
            {"title": tip.title, "description": tip.description, "tips": tip.tips}
            for tip in tips
        ]

        return json.dumps({"category": category, "wellness_tips": result}, indent=2)


class BMICalculatorInput(BaseModel):
    """Input for BMI calculator."""

    weight_kg: float = Field(description="Weight in kilograms")
    height_cm: float = Field(description="Height in centimeters")


class BMICalculatorTool(BaseTool):
    """Tool for calculating BMI."""

    name: str = "bmi_calculator"
    description: str = "Calculate Body Mass Index (BMI) from weight and height"
    args_schema: type[BaseModel] = BMICalculatorInput

    def _run(self, weight_kg: float, height_cm: float) -> str:
        if weight_kg <= 0 or height_cm <= 0:
            return json.dumps({"error": "Weight and height must be positive numbers"})

        height_m = height_cm / 100
        bmi = weight_kg / (height_m**2)

        if bmi < 18.5:
            category = "Underweight"
            advice = (
                "Consider consulting a healthcare provider about healthy weight gain strategies."
            )
        elif bmi < 25:
            category = "Normal weight"
            advice = "Maintain your healthy weight through balanced diet and regular exercise."
        elif bmi < 30:
            category = "Overweight"
            advice = (
                "Consider lifestyle modifications. "
                "Consult a healthcare provider for personalized advice."
            )
        else:
            category = "Obese"
            advice = "Consult a healthcare provider for guidance on weight management."

        return json.dumps(
            {
                "bmi": round(bmi, 1),
                "category": category,
                "healthy_range": "18.5 - 24.9",
                "advice": advice,
                "note": (
                    "BMI is a screening tool but doesn't directly measure body fat or health. "
                    "Other factors matter too."
                ),
                "disclaimer": "Consult a healthcare provider for comprehensive health assessment.",
            },
            indent=2,
        )


class WaterIntakeInput(BaseModel):
    """Input for water intake calculator."""

    weight_kg: float = Field(description="Weight in kilograms")
    activity_level: str = Field(
        default="moderate", description="Activity level: low, moderate, high"
    )


class WaterIntakeTool(BaseTool):
    """Tool for calculating recommended water intake."""

    name: str = "water_intake"
    description: str = "Calculate recommended daily water intake"
    args_schema: type[BaseModel] = WaterIntakeInput

    def _run(self, weight_kg: float, activity_level: str = "moderate") -> str:
        if weight_kg <= 0:
            return json.dumps({"error": "Weight must be positive"})

        base_ml = weight_kg * 33

        multipliers = {"low": 0.9, "moderate": 1.0, "high": 1.3}
        multiplier = multipliers.get(activity_level, 1.0)

        daily_ml = base_ml * multiplier
        daily_liters = daily_ml / 1000
        glasses = daily_ml / 250

        return json.dumps(
            {
                "recommended_daily_ml": round(daily_ml),
                "recommended_daily_liters": round(daily_liters, 1),
                "approximate_glasses": round(glasses),
                "activity_level": activity_level,
                "tips": [
                    "Drink water throughout the day, not all at once",
                    "Increase intake during hot weather or exercise",
                    "Listen to your body's thirst signals",
                    "Urine color should be light yellow",
                ],
                "note": "Individual needs vary. Consult a healthcare provider for personalized advice.",
            },
            indent=2,
        )


class HealthRiskAssessmentInput(BaseModel):
    """Input for health risk assessment."""

    age: int = Field(description="Age in years")
    smoking: bool = Field(default=False, description="Current smoker")
    exercise_days_per_week: int = Field(default=0, description="Days of exercise per week")
    fruit_veg_servings: int = Field(
        default=0, description="Daily fruit and vegetable servings"
    )
    sleep_hours: float = Field(default=7, description="Average hours of sleep per night")


class HealthRiskAssessmentTool(BaseTool):
    """Tool for general health risk assessment."""

    name: str = "health_risk_assessment"
    description: str = "Get a general health risk assessment based on lifestyle factors"
    args_schema: type[BaseModel] = HealthRiskAssessmentInput

    def _run(
        self,
        age: int,
        smoking: bool = False,
        exercise_days_per_week: int = 0,
        fruit_veg_servings: int = 0,
        sleep_hours: float = 7,
    ) -> str:
        score = 100
        recommendations = []

        if smoking:
            score -= 20
            recommendations.append(
                {
                    "area": "Smoking",
                    "status": "high_risk",
                    "advice": (
                        "Quitting smoking is the single best thing you can do for your health. "
                        "Consider cessation programs."
                    ),
                }
            )

        if exercise_days_per_week < 3:
            score -= 15
            recommendations.append(
                {
                    "area": "Physical Activity",
                    "status": "needs_improvement",
                    "advice": (
                        "Aim for at least 150 minutes of moderate exercise per week "
                        "(about 30 min, 5 days)."
                    ),
                }
            )
        elif exercise_days_per_week >= 5:
            recommendations.append(
                {
                    "area": "Physical Activity",
                    "status": "good",
                    "advice": "Great job staying active! Keep it up.",
                }
            )

        if fruit_veg_servings < 5:
            score -= 10
            recommendations.append(
                {
                    "area": "Nutrition",
                    "status": "needs_improvement",
                    "advice": "Aim for 5+ servings of fruits and vegetables daily.",
                }
            )

        if sleep_hours < 6 or sleep_hours > 9:
            score -= 10
            recommendations.append(
                {
                    "area": "Sleep",
                    "status": "needs_improvement",
                    "advice": "Adults should aim for 7-9 hours of quality sleep per night.",
                }
            )
        else:
            recommendations.append(
                {
                    "area": "Sleep",
                    "status": "good",
                    "advice": "Good sleep duration! Focus on sleep quality too.",
                }
            )

        screenings: list[str] = []
        if age >= 40:
            screenings.append("Blood pressure check annually")
            screenings.append("Diabetes screening")
        if age >= 45:
            screenings.append("Colorectal cancer screening")
        if age >= 50:
            screenings.append("Regular health screenings become more important")

        if score >= 80:
            overall = "good"
        elif score >= 60:
            overall = "moderate"
        else:
            overall = "needs_attention"

        return json.dumps(
            {
                "health_score": score,
                "overall_status": overall,
                "recommendations": recommendations,
                "suggested_screenings": screenings or ["Regular check-ups with your doctor"],
                "disclaimer": (
                    "This is a general assessment only. "
                    "Consult healthcare providers for comprehensive evaluation."
                ),
            },
            indent=2,
        )


class EmergencyGuidanceInput(BaseModel):
    """Input for emergency guidance."""

    situation: str = Field(description="Description of the emergency situation")


class EmergencyGuidanceTool(BaseTool):
    """Tool for emergency situation guidance."""

    name: str = "emergency_guidance"
    description: str = "Get guidance for potential emergency health situations"
    args_schema: type[BaseModel] = EmergencyGuidanceInput

    def _run(self, situation: str) -> str:
        situation_lower = situation.lower()

        emergencies = {
            "chest pain": {
                "action": "Call emergency services (911) immediately",
                "symptoms": [
                    "Chest pain or pressure",
                    "Shortness of breath",
                    "Pain radiating to arm/jaw",
                ],
                "do": [
                    "Call 911",
                    "Chew aspirin if not allergic",
                    "Stay calm",
                    "Sit or lie down",
                ],
                "dont": ["Drive yourself", "Ignore symptoms", "Delay calling for help"],
            },
            "stroke": {
                "action": "Call emergency services (911) immediately. Remember FAST.",
                "symptoms": [
                    "Face drooping",
                    "Arm weakness",
                    "Speech difficulty",
                    "Time to call 911",
                ],
                "do": [
                    "Call 911 immediately",
                    "Note time symptoms started",
                    "Keep person comfortable",
                ],
                "dont": ["Give food or water", "Let them fall asleep", "Delay"],
            },
            "choking": {
                "action": "Perform Heimlich maneuver if trained, call 911",
                "symptoms": [
                    "Unable to speak or cough",
                    "Clutching throat",
                    "Turning blue",
                ],
                "do": [
                    "Stand behind person",
                    "Give abdominal thrusts",
                    "Call 911 if not resolved",
                ],
                "dont": ["Give water", "Slap back if partially choking", "Leave alone"],
            },
            "allergic reaction": {
                "action": "Use epinephrine if available, call 911 for severe reactions",
                "symptoms": ["Swelling", "Difficulty breathing", "Hives", "Dizziness"],
                "do": [
                    "Use EpiPen if available",
                    "Call 911",
                    "Help person lie down",
                    "Loosen tight clothing",
                ],
                "dont": ["Leave alone", "Give oral medication if swallowing difficult"],
            },
        }

        for emergency_type, info in emergencies.items():
            if emergency_type in situation_lower:
                return json.dumps(
                    {
                        "urgency": "EMERGENCY",
                        "immediate_action": info["action"],
                        "warning_signs": info["symptoms"],
                        "what_to_do": info["do"],
                        "what_not_to_do": info["dont"],
                        "important": "If in doubt, always call emergency services (911).",
                    },
                    indent=2,
                )

        return json.dumps(
            {
                "urgency": "ASSESS",
                "guidance": (
                    "If you suspect a medical emergency, "
                    "call emergency services (911) immediately."
                ),
                "emergency_signs": [
                    "Difficulty breathing",
                    "Chest pain",
                    "Severe bleeding",
                    "Loss of consciousness",
                    "Severe allergic reaction",
                    "Signs of stroke (FAST)",
                    "Severe burns",
                    "Poisoning",
                ],
                "action": "When in doubt, it's always better to seek emergency help.",
                "important": "Do not use this app to handle medical emergencies. Call 911 immediately.",
            },
            indent=2,
        )


def get_health_tools() -> list[BaseTool]:
    """Get all health assistant tools."""
    return [
        SymptomCheckerTool(),
        ConditionInfoTool(),
        WellnessTipsTool(),
        BMICalculatorTool(),
        WaterIntakeTool(),
        HealthRiskAssessmentTool(),
        EmergencyGuidanceTool(),
    ]
