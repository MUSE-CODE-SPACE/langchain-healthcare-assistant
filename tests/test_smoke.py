"""Smoke tests for the healthcare assistant.

These tests deliberately avoid any LLM calls (``LLM_PROVIDER=none`` is set in
``conftest.py``) so they run quickly and offline in CI.
"""

from __future__ import annotations

import json

from app.agents.health_agent import HealthAssistantAgent
from app.tools.health_tools import BMICalculatorTool


def test_app_starts():
    from app.api import app, create_app

    assert app is not None
    fresh = create_app()
    assert fresh.name == "app.api"


def test_health_endpoint(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    body = res.get_json()
    assert body["status"] == "healthy"
    assert body["service"] == "healthcare-assistant"
    assert "timestamp" in body


def test_chat_keyword_fallback(client):
    """With no LLM keys, the keyword router must respond successfully."""
    res = client.post(
        "/api/chat",
        data=json.dumps({"message": "give me some nutrition tips"}),
        content_type="application/json",
    )
    assert res.status_code == 200
    body = res.get_json()
    assert "response" in body
    assert body["mode"] == "none"
    assert len(body["response"]) > 50


def test_chat_requires_message(client):
    res = client.post("/api/chat", json={})
    assert res.status_code == 400


def test_emergency_detection_chest_pain(client):
    res = client.post("/api/chat", json={"message": "I'm having chest pain"})
    assert res.status_code == 200
    body = res.get_json()
    text = body["response"]
    assert "EMERGENCY" in text
    assert "911" in text


def test_emergency_detection_suicide():
    agent = HealthAssistantAgent()
    response = agent.chat("I am thinking about suicide")
    assert "EMERGENCY" in response
    assert "911" in response or "988" in response


def test_emergency_detection_cant_breathe():
    agent = HealthAssistantAgent()
    response = agent.chat("I can't breathe")
    assert "EMERGENCY" in response


def test_symptom_lookup_keyword_router():
    agent = HealthAssistantAgent()
    response = agent.chat("I have a headache")
    assert "Headache" in response
    # Disclaimer must be present on medical responses.
    assert "DISCLAIMER" in response


def test_bmi_tool_accuracy():
    tool = BMICalculatorTool()
    result = json.loads(tool._run(weight_kg=70, height_cm=175))
    # 70 / (1.75^2) = 22.857 → rounded to 22.9
    assert result["bmi"] == 22.9
    assert result["category"] == "Normal weight"


def test_bmi_tool_categories():
    tool = BMICalculatorTool()
    underweight = json.loads(tool._run(weight_kg=45, height_cm=175))
    assert underweight["category"] == "Underweight"

    overweight = json.loads(tool._run(weight_kg=85, height_cm=175))
    assert overweight["category"] == "Overweight"

    obese = json.loads(tool._run(weight_kg=100, height_cm=170))
    assert obese["category"] == "Obese"


def test_bmi_tool_rejects_invalid_input():
    tool = BMICalculatorTool()
    result = json.loads(tool._run(weight_kg=0, height_cm=175))
    assert "error" in result


def test_list_symptoms_endpoint(client):
    res = client.get("/api/symptoms")
    assert res.status_code == 200
    body = res.get_json()
    assert "symptoms" in body
    assert len(body["symptoms"]) > 0
    assert any(s["id"] == "headache" for s in body["symptoms"])


def test_session_reset(client):
    res = client.post("/api/session/reset", json={"session_id": "demo"})
    assert res.status_code == 200
    body = res.get_json()
    assert body["status"] == "reset"
    assert body["session_id"] == "demo"
