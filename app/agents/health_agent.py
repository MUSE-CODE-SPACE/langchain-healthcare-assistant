"""
Health Agent - LangChain agent for health assistance
DISCLAIMER: For informational purposes only. Not medical advice.
"""

import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory

from app.tools.health_tools import get_health_tools, DISCLAIMER
from app.data.health_knowledge import (
    SYMPTOMS_DB, CONDITIONS_DB, WELLNESS_TIPS,
    get_symptom, get_condition, search_symptoms
)


@dataclass
class HealthContext:
    """Maintains health consultation context."""
    user_id: str = "default"
    reported_symptoms: List[str] = field(default_factory=list)
    discussed_conditions: List[str] = field(default_factory=list)
    wellness_interests: List[str] = field(default_factory=list)
    bmi_data: Optional[Dict] = None
    risk_assessment: Optional[Dict] = None


class HealthAssistantAgent:
    """
    Health information assistant that can:
    - Provide general symptom information
    - Explain health conditions
    - Give wellness tips
    - Calculate BMI and water intake
    - Assess general health risks
    - Guide during emergencies (always recommends calling 911)

    IMPORTANT: This is NOT a substitute for professional medical advice.
    """

    SYSTEM_PROMPT = """You are a helpful health information assistant.

IMPORTANT DISCLAIMER: You provide general health information for educational purposes only.
You are NOT a doctor and do NOT provide medical diagnoses, treatment recommendations, or medical advice.
Always encourage users to consult qualified healthcare professionals for medical concerns.

Your role:
1. **Educate**: Provide general health information
2. **Inform**: Share wellness tips and healthy lifestyle guidance
3. **Guide**: Help users understand when to seek medical attention
4. **Support**: Offer empathetic, helpful responses
5. **Redirect**: Always recommend professional care for medical concerns

Guidelines:
- Never diagnose conditions
- Never recommend specific treatments or medications
- Always include disclaimers with medical information
- Encourage users to seek professional medical advice
- Be empathetic but clear about your limitations
- In emergencies, immediately advise calling emergency services

You have access to tools for symptom information, wellness tips, and health calculators."""

    def __init__(self, llm=None, verbose: bool = False):
        """Initialize the health assistant agent."""
        self.llm = llm
        self.verbose = verbose
        self.context = HealthContext()
        self.conversation_history: List[Dict[str, str]] = []
        self.tools = get_health_tools()

        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=10
        )

    def chat(self, user_message: str) -> str:
        """Process user message and generate response."""

        # Add to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message,
            "timestamp": datetime.now().isoformat()
        })

        # Check for emergency keywords first
        if self._is_emergency(user_message):
            response = self._handle_emergency(user_message)
        else:
            response = self._generate_response(user_message)

        # Add response to history
        self.conversation_history.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now().isoformat()
        })

        return response

    def _is_emergency(self, message: str) -> bool:
        """Check if message indicates emergency."""
        emergency_keywords = [
            'heart attack', 'stroke', 'can\'t breathe', 'choking',
            'severe bleeding', 'unconscious', 'overdose', 'suicide',
            'chest pain', 'anaphylaxis', 'seizure'
        ]
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in emergency_keywords)

    def _handle_emergency(self, message: str) -> str:
        """Handle potential emergency situations."""
        response = """# EMERGENCY ALERT

**If this is a medical emergency, call 911 (or your local emergency number) IMMEDIATELY.**

While waiting for help:
- Stay calm
- Do not leave the person alone
- Follow dispatcher instructions
- If trained, provide first aid

**Common Emergency Signs:**
- Difficulty breathing
- Chest pain or pressure
- Severe bleeding
- Loss of consciousness
- Signs of stroke (FAST: Face drooping, Arm weakness, Speech difficulty, Time to call 911)

**I am an AI assistant and CANNOT provide emergency medical care.** Please call emergency services immediately for any life-threatening situation.

---

"""
        response += DISCLAIMER
        return response

    def _generate_response(self, message: str) -> str:
        """Generate appropriate health response."""
        message_lower = message.lower()

        # Check for symptom queries
        if any(word in message_lower for word in ['symptom', 'have', 'feeling', 'pain', 'ache', 'hurt']):
            return self._handle_symptom_query(message)

        # Check for condition queries
        if any(word in message_lower for word in ['what is', 'tell me about', 'condition', 'disease', 'illness']):
            return self._handle_condition_query(message)

        # Check for wellness tips
        if any(word in message_lower for word in ['tip', 'advice', 'healthy', 'wellness', 'lifestyle', 'improve']):
            return self._handle_wellness_query(message)

        # Check for calculators
        if any(word in message_lower for word in ['bmi', 'weight', 'body mass']):
            return self._handle_bmi_query(message)

        if any(word in message_lower for word in ['water', 'hydration', 'drink']):
            return self._handle_hydration_query(message)

        # Check for risk assessment
        if any(word in message_lower for word in ['risk', 'assessment', 'evaluate', 'check my health']):
            return self._handle_risk_query(message)

        # Default response
        return self._handle_general_query(message)

    def _handle_symptom_query(self, message: str) -> str:
        """Handle symptom-related queries."""
        # Try to identify symptom
        symptom_found = None
        for symptom_key, symptom in SYMPTOMS_DB.items():
            if symptom_key in message.lower() or symptom.name.lower() in message.lower():
                symptom_found = symptom
                self.context.reported_symptoms.append(symptom.name)
                break

        if not symptom_found:
            # Search for related symptoms
            results = search_symptoms(message)
            if results:
                symptom_found = results[0]

        if symptom_found:
            response = f"# Information about {symptom_found.name}\n\n"
            response += f"**Description:** {symptom_found.description}\n\n"

            response += "**Common Causes:**\n"
            for cause in symptom_found.common_causes[:5]:
                response += f"- {cause}\n"

            response += "\n**General Home Care Tips:**\n"
            for remedy in symptom_found.home_remedies[:4]:
                response += f"- {remedy}\n"

            response += "\n**When to See a Doctor:**\n"
            for sign in symptom_found.when_to_see_doctor[:4]:
                response += f"- {sign}\n"

            response += "\n---\n"
            response += DISCLAIMER
            return response

        return self._symptom_not_found_response()

    def _symptom_not_found_response(self) -> str:
        """Response when symptom not found."""
        response = """# Health Information Request

I couldn't find specific information about that symptom in my database.

**For any health concerns, I recommend:**
- Consulting with a healthcare provider
- Keeping track of when symptoms started and their severity
- Noting any other associated symptoms
- Being prepared to describe your medical history

**I can provide general information about:**
- Common symptoms (headache, fatigue, sore throat, back pain, etc.)
- Health conditions
- Wellness and lifestyle tips
- BMI and hydration calculations

---

"""
        response += DISCLAIMER
        return response

    def _handle_condition_query(self, message: str) -> str:
        """Handle condition-related queries."""
        condition_found = None
        for cond_key, condition in CONDITIONS_DB.items():
            if cond_key in message.lower() or condition.name.lower() in message.lower():
                condition_found = condition
                self.context.discussed_conditions.append(condition.name)
                break

        if condition_found:
            response = f"# About {condition_found.name}\n\n"
            response += f"**Overview:** {condition_found.description}\n\n"

            response += "**Common Symptoms:**\n"
            for symptom in condition_found.common_symptoms:
                response += f"- {symptom}\n"

            response += "\n**Risk Factors:**\n"
            for factor in condition_found.risk_factors[:5]:
                response += f"- {factor}\n"

            response += "\n**Prevention:**\n"
            for prev in condition_found.prevention[:4]:
                response += f"- {prev}\n"

            response += f"\n**General Information:** {condition_found.treatment_overview}\n"

            response += "\n**Seek Medical Help If:**\n"
            for sign in condition_found.when_to_seek_help[:3]:
                response += f"- {sign}\n"

            response += "\n---\n"
            response += DISCLAIMER
            return response

        return self._condition_not_found_response()

    def _condition_not_found_response(self) -> str:
        """Response when condition not found."""
        response = """# Health Condition Information

I don't have detailed information about that specific condition.

**Conditions I can provide general information about:**
- Common Cold
- Influenza (Flu)
- Allergies
- High Blood Pressure (Hypertension)
- Type 2 Diabetes

**For specific conditions:**
- Consult your healthcare provider
- Visit reputable medical websites
- Check with your pharmacist for general questions

---

"""
        response += DISCLAIMER
        return response

    def _handle_wellness_query(self, message: str) -> str:
        """Handle wellness and lifestyle queries."""
        message_lower = message.lower()

        # Identify category
        category = "general"
        if any(word in message_lower for word in ['nutrition', 'diet', 'food', 'eating']):
            category = "nutrition"
        elif any(word in message_lower for word in ['exercise', 'fitness', 'workout', 'active']):
            category = "exercise"
        elif any(word in message_lower for word in ['sleep', 'rest', 'insomnia']):
            category = "sleep"
        elif any(word in message_lower for word in ['mental', 'stress', 'anxiety', 'mood']):
            category = "mental_health"
        elif any(word in message_lower for word in ['prevention', 'screening', 'checkup']):
            category = "preventive_care"

        response = f"# Wellness Tips: {category.replace('_', ' ').title()}\n\n"

        for tip in WELLNESS_TIPS:
            if tip.category == category or category == "general":
                response += f"## {tip.title}\n"
                response += f"{tip.description}\n\n"
                for t in tip.tips[:6]:
                    response += f"- {t}\n"
                response += "\n"

                if category != "general":
                    break

        response += "---\n"
        response += "Remember: Small, consistent changes lead to lasting improvements!\n"
        return response

    def _handle_bmi_query(self, message: str) -> str:
        """Handle BMI-related queries."""
        response = """# BMI Calculator

**Body Mass Index (BMI)** is a screening tool that can indicate whether a person has a healthy body weight.

**To calculate your BMI, I need:**
- Your weight in kilograms (kg)
- Your height in centimeters (cm)

**Example:** "Calculate BMI for 70 kg and 175 cm"

**BMI Categories:**
- Under 18.5: Underweight
- 18.5 - 24.9: Normal weight
- 25 - 29.9: Overweight
- 30 and above: Obese

**Note:** BMI doesn't directly measure body fat and doesn't account for muscle mass, age, or gender differences. It's just one factor in assessing health.

---

"""
        response += DISCLAIMER
        return response

    def _handle_hydration_query(self, message: str) -> str:
        """Handle hydration-related queries."""
        response = """# Hydration Guide

**Staying hydrated is essential for:**
- Regulating body temperature
- Maintaining joint health
- Transporting nutrients
- Supporting organ function

**General Recommendations:**
- Adults: Approximately 8 glasses (2 liters) per day
- Increase intake during exercise, hot weather, or illness
- Listen to your body's thirst signals

**Signs of Dehydration:**
- Dark urine
- Dry mouth
- Fatigue
- Headache
- Dizziness

**Tips for Better Hydration:**
- Carry a water bottle
- Set reminders to drink water
- Eat water-rich foods (fruits, vegetables)
- Drink a glass of water with each meal

---

"""
        response += "Individual needs vary based on weight, activity level, and climate.\n"
        return response

    def _handle_risk_query(self, message: str) -> str:
        """Handle health risk assessment queries."""
        response = """# General Health Risk Assessment

I can provide a basic lifestyle health assessment based on:
- Age
- Smoking status
- Exercise frequency
- Nutrition habits
- Sleep patterns

**To get your assessment, share:**
- Your age
- Whether you smoke
- Days of exercise per week
- Daily fruit/vegetable servings
- Average hours of sleep

**Example:** "I'm 35, don't smoke, exercise 3 days/week, eat 4 servings of fruits/vegetables, sleep 7 hours"

**Important:** This is a general wellness assessment only. It does NOT replace professional medical evaluation.

---

"""
        response += DISCLAIMER
        return response

    def _handle_general_query(self, message: str) -> str:
        """Handle general health queries."""
        response = """# Health Assistant

Hello! I'm your health information assistant. I can help with:

**General Health Information:**
- Symptom information and when to see a doctor
- Health condition overviews
- Wellness and lifestyle tips

**Health Calculators:**
- BMI calculator
- Water intake recommendations
- General health risk assessment

**Wellness Topics:**
- Nutrition and diet
- Exercise and fitness
- Sleep hygiene
- Mental wellness
- Preventive care

**What would you like to know about?**

---

**Remember:** I provide general information only. For medical advice, diagnosis, or treatment, please consult a qualified healthcare professional.
"""
        return response

    def reset(self):
        """Reset the health assistant."""
        self.context = HealthContext()
        self.conversation_history = []
        self.memory.clear()


def create_health_agent(llm=None, verbose: bool = False) -> HealthAssistantAgent:
    """Factory function to create health assistant agent."""
    return HealthAssistantAgent(llm=llm, verbose=verbose)
