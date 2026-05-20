"""
Health Agent - LangChain 0.3 agent for the healthcare assistant.

DISCLAIMER: This module powers an *informational* assistant. It is NOT a
substitute for professional medical advice, diagnosis, or treatment.

The agent supports three modes, chosen automatically (or via the
``LLM_PROVIDER`` environment variable):

- ``anthropic`` — uses ChatAnthropic (Claude) when ``ANTHROPIC_API_KEY`` is set
- ``openai`` — uses ChatOpenAI (GPT) when ``OPENAI_API_KEY`` is set
- ``none`` — graceful fallback to a deterministic keyword router (no network)

Regardless of mode, an emergency-keyword pre-filter always runs *first* so
that high-risk inputs (chest pain, suicide, can't breathe, etc.) are routed
to a strict emergency response that points the user to 911.
"""

from __future__ import annotations

import logging
import os
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from app.data.health_knowledge import (
    CONDITIONS_DB,
    SYMPTOMS_DB,
    WELLNESS_TIPS,
    search_symptoms,
)
from app.tools.health_tools import DISCLAIMER, get_health_tools

logger = logging.getLogger(__name__)


# Emergency keywords are checked BEFORE any LLM call. Keep this list tight
# (high-precision) so it does not crowd out ordinary health questions, but
# broad enough to catch the obvious life-threatening cues.
EMERGENCY_KEYWORDS: tuple[str, ...] = (
    "heart attack",
    "stroke",
    "can't breathe",
    "cannot breathe",
    "can not breathe",
    "not breathing",
    "choking",
    "severe bleeding",
    "bleeding heavily",
    "unconscious",
    "overdose",
    "suicide",
    "kill myself",
    "chest pain",
    "anaphylaxis",
    "seizure",
)


SYSTEM_PROMPT = """You are a careful, empathetic *health information* assistant.

CRITICAL DISCLAIMER (read every turn):
- You are NOT a doctor and you do NOT provide medical diagnoses, treatment plans,
  prescriptions, or personalized medical advice.
- The information you share is GENERAL and EDUCATIONAL only.
- You ALWAYS recommend that the user consult a qualified healthcare professional
  for anything beyond general information.
- For any sign of a medical emergency (chest pain, trouble breathing, stroke
  symptoms, severe bleeding, suicidal ideation, overdose, anaphylaxis,
  seizure, loss of consciousness), you IMMEDIATELY direct the user to call
  their local emergency number (911 in the U.S.) and do NOT attempt to manage
  the situation yourself.

How you should respond:
1. Be warm, plain-spoken, and concise.
2. Use the provided tools when they fit (symptom_checker, condition_info,
   wellness_tips, bmi_calculator, water_intake, health_risk_assessment,
   emergency_guidance). Prefer tools over free-form recall for factual lookups.
3. Always end medically-themed responses with a short reminder that this is
   educational information, not a substitute for professional care.
4. Never recommend specific prescription medications or dosages.
5. If the user describes symptoms that *could* be serious, encourage them to
   contact a clinician promptly and describe red-flag signs to watch for.
"""


@dataclass
class HealthContext:
    """Maintains lightweight health consultation context for a single session."""

    user_id: str = "default"
    reported_symptoms: list[str] = field(default_factory=list)
    discussed_conditions: list[str] = field(default_factory=list)
    wellness_interests: list[str] = field(default_factory=list)
    bmi_data: dict[str, Any] | None = None
    risk_assessment: dict[str, Any] | None = None


def _select_llm() -> tuple[str, Any | None]:
    """Pick an LLM backend based on env vars.

    Returns a tuple ``(mode, llm)``. ``mode`` is one of
    ``"anthropic" | "openai" | "none"`` and ``llm`` is the configured chat
    model or ``None`` for the keyword-router fallback.
    """

    provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))

    # Explicit "none" disables LLM mode even when keys are present (useful in tests / CI).
    if provider == "none":
        return "none", None

    # Auto-detect when no explicit provider was set.
    if not provider:
        if has_anthropic:
            provider = "anthropic"
        elif has_openai:
            provider = "openai"
        else:
            return "none", None

    try:
        if provider == "anthropic":
            if not has_anthropic:
                logger.warning("LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is missing; falling back.")
                return "none", None
            from langchain_anthropic import ChatAnthropic

            model = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
            return "anthropic", ChatAnthropic(model=model, temperature=0.2, max_tokens=1024)

        if provider == "openai":
            if not has_openai:
                logger.warning("LLM_PROVIDER=openai but OPENAI_API_KEY is missing; falling back.")
                return "none", None
            from langchain_openai import ChatOpenAI

            model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
            return "openai", ChatOpenAI(model=model, temperature=0.2)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to initialise LLM provider %s: %s; using keyword fallback.", provider, exc)
        return "none", None

    return "none", None


class HealthAssistantAgent:
    """
    Health information assistant.

    Capabilities:
    - Provide general symptom and condition information
    - Share wellness and lifestyle guidance
    - Compute BMI / water intake / lifestyle risk score
    - Recognise emergency keywords and steer the user to 911

    IMPORTANT: This is NOT a substitute for professional medical advice.
    """

    SYSTEM_PROMPT = SYSTEM_PROMPT

    def __init__(
        self,
        llm: Any | None = None,
        verbose: bool = False,
        history_size: int = 10,
    ) -> None:
        self.verbose = verbose
        self.context = HealthContext()
        # ``conversation_history`` retains the full transcript for inspection;
        # ``recent_messages`` is the windowed buffer used as LCEL chat history.
        self.conversation_history: list[dict[str, str]] = []
        self.recent_messages: deque[tuple[str, str]] = deque(maxlen=history_size * 2)
        self.tools = get_health_tools()

        if llm is not None:
            self.llm = llm
            # Best-effort mode label when caller injects a custom LLM.
            self.mode = "custom"
        else:
            self.mode, self.llm = _select_llm()

        self._executor = self._build_executor() if self.llm is not None else None

    # ------------------------------------------------------------------ public

    def chat(self, user_message: str) -> str:
        """Process a user message and return the assistant response."""
        self._append_history("user", user_message)

        # Emergency pre-filter runs BEFORE any LLM call.
        if self._is_emergency(user_message):
            response = self._handle_emergency(user_message)
        elif self._executor is not None:
            response = self._llm_response(user_message)
        else:
            response = self._keyword_router(user_message)

        self._append_history("assistant", response)
        return response

    def reset(self) -> None:
        """Reset session state."""
        self.context = HealthContext()
        self.conversation_history = []
        self.recent_messages.clear()

    # ------------------------------------------------------------------ internals

    def _append_history(self, role: str, content: str) -> None:
        self.conversation_history.append(
            {
                "role": role,
                "content": content,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )
        self.recent_messages.append((role, content))

    def _build_executor(self) -> Any | None:
        """Wire up a tool-calling agent + executor using LangChain 0.3 LCEL."""
        try:
            from langchain.agents import AgentExecutor, create_tool_calling_agent
            from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", self.SYSTEM_PROMPT),
                    MessagesPlaceholder("chat_history", optional=True),
                    ("human", "{input}"),
                    MessagesPlaceholder("agent_scratchpad"),
                ]
            )
            agent = create_tool_calling_agent(self.llm, self.tools, prompt)
            return AgentExecutor(
                agent=agent,
                tools=self.tools,
                verbose=self.verbose,
                handle_parsing_errors=True,
                max_iterations=4,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not build agent executor: %s; using keyword fallback.", exc)
            return None

    def _llm_response(self, user_message: str) -> str:
        assert self._executor is not None
        try:
            history_msgs = self._chat_history_for_lcel()
            result = self._executor.invoke(
                {"input": user_message, "chat_history": history_msgs}
            )
            answer = result.get("output") if isinstance(result, dict) else str(result)
            answer = answer or ""
            # Always append the disclaimer if the model omitted it.
            if "DISCLAIMER" not in answer and "disclaimer" not in answer.lower():
                answer = f"{answer.rstrip()}\n\n---\n{DISCLAIMER}"
            return answer
        except Exception as exc:
            logger.warning("LLM execution failed: %s; falling back to keyword router.", exc)
            return self._keyword_router(user_message)

    def _chat_history_for_lcel(self) -> list[Any]:
        """Convert the deque into LangChain message objects."""
        try:
            from langchain_core.messages import AIMessage, HumanMessage
        except Exception:  # pragma: no cover
            return []

        messages: list[Any] = []
        # Skip the just-appended user message (it goes into "input").
        for role, content in list(self.recent_messages)[:-1]:
            if role == "user":
                messages.append(HumanMessage(content=content))
            else:
                messages.append(AIMessage(content=content))
        return messages

    # -- emergency handling -------------------------------------------------

    @staticmethod
    def _is_emergency(message: str) -> bool:
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in EMERGENCY_KEYWORDS)

    @staticmethod
    def _handle_emergency(_message: str) -> str:
        return (
            "# EMERGENCY ALERT\n\n"
            "**If this is a medical emergency, call 911 (or your local emergency "
            "number) IMMEDIATELY.**\n\n"
            "While waiting for help:\n"
            "- Stay calm\n"
            "- Do not leave the person alone\n"
            "- Follow dispatcher instructions\n"
            "- If trained, provide first aid\n\n"
            "**Common Emergency Signs:**\n"
            "- Difficulty breathing\n"
            "- Chest pain or pressure\n"
            "- Severe bleeding\n"
            "- Loss of consciousness\n"
            "- Signs of stroke (FAST: Face drooping, Arm weakness, "
            "Speech difficulty, Time to call 911)\n"
            "- Suicidal thoughts — in the U.S., dial or text 988 (Suicide & "
            "Crisis Lifeline)\n\n"
            "**I am an AI assistant and CANNOT provide emergency medical care.** "
            "Please call emergency services immediately for any life-threatening "
            "situation.\n\n---\n" + DISCLAIMER
        )

    # -- keyword router (fallback) ----------------------------------------

    def _keyword_router(self, message: str) -> str:
        message_lower = message.lower()

        if any(word in message_lower for word in ("symptom", "have", "feeling", "pain", "ache", "hurt")):
            return self._handle_symptom_query(message)

        if any(
            word in message_lower
            for word in ("what is", "tell me about", "condition", "disease", "illness")
        ):
            return self._handle_condition_query(message)

        if any(
            word in message_lower
            for word in ("tip", "advice", "healthy", "wellness", "lifestyle", "improve")
        ):
            return self._handle_wellness_query(message)

        if any(word in message_lower for word in ("bmi", "weight", "body mass")):
            return self._handle_bmi_query(message)

        if any(word in message_lower for word in ("water", "hydration", "drink")):
            return self._handle_hydration_query(message)

        if any(
            word in message_lower
            for word in ("risk", "assessment", "evaluate", "check my health")
        ):
            return self._handle_risk_query(message)

        return self._handle_general_query(message)

    def _handle_symptom_query(self, message: str) -> str:
        symptom_found = None
        for symptom_key, symptom in SYMPTOMS_DB.items():
            if symptom_key in message.lower() or symptom.name.lower() in message.lower():
                symptom_found = symptom
                self.context.reported_symptoms.append(symptom.name)
                break

        if not symptom_found:
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

    @staticmethod
    def _symptom_not_found_response() -> str:
        return (
            "# Health Information Request\n\n"
            "I couldn't find specific information about that symptom in my database.\n\n"
            "**For any health concerns, I recommend:**\n"
            "- Consulting with a healthcare provider\n"
            "- Keeping track of when symptoms started and their severity\n"
            "- Noting any other associated symptoms\n"
            "- Being prepared to describe your medical history\n\n"
            "**I can provide general information about:**\n"
            "- Common symptoms (headache, fatigue, sore throat, back pain, etc.)\n"
            "- Health conditions\n"
            "- Wellness and lifestyle tips\n"
            "- BMI and hydration calculations\n\n---\n" + DISCLAIMER
        )

    def _handle_condition_query(self, message: str) -> str:
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

    @staticmethod
    def _condition_not_found_response() -> str:
        return (
            "# Health Condition Information\n\n"
            "I don't have detailed information about that specific condition.\n\n"
            "**Conditions I can provide general information about:**\n"
            "- Common Cold\n"
            "- Influenza (Flu)\n"
            "- Allergies\n"
            "- High Blood Pressure (Hypertension)\n"
            "- Type 2 Diabetes\n\n"
            "**For specific conditions:**\n"
            "- Consult your healthcare provider\n"
            "- Visit reputable medical websites\n"
            "- Check with your pharmacist for general questions\n\n---\n" + DISCLAIMER
        )

    @staticmethod
    def _handle_wellness_query(message: str) -> str:
        message_lower = message.lower()

        category = "general"
        if any(word in message_lower for word in ("nutrition", "diet", "food", "eating")):
            category = "nutrition"
        elif any(word in message_lower for word in ("exercise", "fitness", "workout", "active")):
            category = "exercise"
        elif any(word in message_lower for word in ("sleep", "rest", "insomnia")):
            category = "sleep"
        elif any(word in message_lower for word in ("mental", "stress", "anxiety", "mood")):
            category = "mental_health"
        elif any(word in message_lower for word in ("prevention", "screening", "checkup")):
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

    @staticmethod
    def _handle_bmi_query(_message: str) -> str:
        return (
            "# BMI Calculator\n\n"
            "**Body Mass Index (BMI)** is a screening tool that can indicate "
            "whether a person has a healthy body weight.\n\n"
            "**To calculate your BMI, I need:**\n"
            "- Your weight in kilograms (kg)\n"
            "- Your height in centimeters (cm)\n\n"
            '**Example:** "Calculate BMI for 70 kg and 175 cm"\n\n'
            "**BMI Categories:**\n"
            "- Under 18.5: Underweight\n"
            "- 18.5 - 24.9: Normal weight\n"
            "- 25 - 29.9: Overweight\n"
            "- 30 and above: Obese\n\n"
            "**Note:** BMI doesn't directly measure body fat and doesn't account "
            "for muscle mass, age, or gender differences. It's just one factor in "
            "assessing health.\n\n---\n" + DISCLAIMER
        )

    @staticmethod
    def _handle_hydration_query(_message: str) -> str:
        return (
            "# Hydration Guide\n\n"
            "**Staying hydrated is essential for:**\n"
            "- Regulating body temperature\n"
            "- Maintaining joint health\n"
            "- Transporting nutrients\n"
            "- Supporting organ function\n\n"
            "**General Recommendations:**\n"
            "- Adults: Approximately 8 glasses (2 liters) per day\n"
            "- Increase intake during exercise, hot weather, or illness\n"
            "- Listen to your body's thirst signals\n\n"
            "**Signs of Dehydration:**\n"
            "- Dark urine\n"
            "- Dry mouth\n"
            "- Fatigue\n"
            "- Headache\n"
            "- Dizziness\n\n"
            "**Tips for Better Hydration:**\n"
            "- Carry a water bottle\n"
            "- Set reminders to drink water\n"
            "- Eat water-rich foods (fruits, vegetables)\n"
            "- Drink a glass of water with each meal\n\n---\n"
            "Individual needs vary based on weight, activity level, and climate.\n"
        )

    @staticmethod
    def _handle_risk_query(_message: str) -> str:
        return (
            "# General Health Risk Assessment\n\n"
            "I can provide a basic lifestyle health assessment based on:\n"
            "- Age\n"
            "- Smoking status\n"
            "- Exercise frequency\n"
            "- Nutrition habits\n"
            "- Sleep patterns\n\n"
            "**To get your assessment, share:**\n"
            "- Your age\n"
            "- Whether you smoke\n"
            "- Days of exercise per week\n"
            "- Daily fruit/vegetable servings\n"
            "- Average hours of sleep\n\n"
            "**Example:** \"I'm 35, don't smoke, exercise 3 days/week, eat 4 "
            'servings of fruits/vegetables, sleep 7 hours"\n\n'
            "**Important:** This is a general wellness assessment only. It does "
            "NOT replace professional medical evaluation.\n\n---\n" + DISCLAIMER
        )

    @staticmethod
    def _handle_general_query(_message: str) -> str:
        return (
            "# Health Assistant\n\n"
            "Hello! I'm your health information assistant. I can help with:\n\n"
            "**General Health Information:**\n"
            "- Symptom information and when to see a doctor\n"
            "- Health condition overviews\n"
            "- Wellness and lifestyle tips\n\n"
            "**Health Calculators:**\n"
            "- BMI calculator\n"
            "- Water intake recommendations\n"
            "- General health risk assessment\n\n"
            "**Wellness Topics:**\n"
            "- Nutrition and diet\n"
            "- Exercise and fitness\n"
            "- Sleep hygiene\n"
            "- Mental wellness\n"
            "- Preventive care\n\n"
            "**What would you like to know about?**\n\n---\n"
            "**Remember:** I provide general information only. For medical "
            "advice, diagnosis, or treatment, please consult a qualified "
            "healthcare professional.\n"
        )


def create_health_agent(llm: Any | None = None, verbose: bool = False) -> HealthAssistantAgent:
    """Factory function to create a health assistant agent."""
    return HealthAssistantAgent(llm=llm, verbose=verbose)


# ``Iterable`` re-export keeps mypy quiet when callers annotate factories.
__all__ = [
    "EMERGENCY_KEYWORDS",
    "HealthAssistantAgent",
    "HealthContext",
    "Iterable",
    "create_health_agent",
]
