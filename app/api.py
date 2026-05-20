"""
Flask API for the Healthcare Assistant.

DISCLAIMER: This service exposes a *health information* assistant. It does
NOT provide medical advice. See the README and the in-app disclaimers.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_cors import CORS

from app.agents.health_agent import HealthAssistantAgent, create_health_agent
from app.data.health_knowledge import CONDITIONS_DB, SYMPTOMS_DB, WELLNESS_TIPS

load_dotenv()

logger = logging.getLogger(__name__)

# Repo layout: ``app/api.py`` — templates/static live one level up if present.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_TEMPLATE_DIR = _REPO_ROOT / "templates"
_STATIC_DIR = _REPO_ROOT / "static"
_DOCS_DIR = _REPO_ROOT / "docs"


def create_app() -> Flask:
    app = Flask(
        __name__,
        static_folder=str(_STATIC_DIR) if _STATIC_DIR.exists() else None,
        template_folder=str(_TEMPLATE_DIR) if _TEMPLATE_DIR.exists() else None,
    )
    CORS(app)

    agents: dict[str, HealthAssistantAgent] = {}

    def get_agent(session_id: str) -> HealthAssistantAgent:
        if session_id not in agents:
            agents[session_id] = create_health_agent()
        return agents[session_id]

    @app.route("/")
    def index():
        # Prefer templates/index.html when present, otherwise fall back to docs/.
        if _TEMPLATE_DIR.exists() and (_TEMPLATE_DIR / "index.html").exists():
            return render_template("index.html")
        if (_DOCS_DIR / "index.html").exists():
            return send_from_directory(str(_DOCS_DIR), "index.html")
        return jsonify(
            {
                "service": "healthcare-assistant",
                "message": (
                    "API is running. POST /api/chat with {\"message\": \"...\"} "
                    "to talk to the assistant."
                ),
                "endpoints": [
                    "GET /api/health",
                    "POST /api/chat",
                    "GET /api/symptoms",
                    "GET /api/conditions",
                    "GET /api/wellness",
                    "POST /api/session/reset",
                ],
            }
        )

    @app.route("/static/<path:filename>")
    def serve_static(filename: str):
        if app.static_folder is None:
            return jsonify({"error": "static assets not available"}), 404
        return send_from_directory(app.static_folder, filename)

    @app.route("/api/health", methods=["GET"])
    def health_check():
        return jsonify(
            {
                "status": "healthy",
                "service": "healthcare-assistant",
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )

    @app.route("/api/chat", methods=["POST"])
    def chat():
        data = request.get_json(silent=True) or {}
        session_id = data.get("session_id", "default")
        message = data.get("message", "")

        if not message:
            return jsonify({"error": "Message is required"}), 400

        try:
            agent = get_agent(session_id)
            response = agent.chat(message)
            return jsonify(
                {
                    "response": response,
                    "session_id": session_id,
                    "mode": getattr(agent, "mode", "unknown"),
                }
            )
        except Exception as exc:
            logger.exception("Chat request failed")
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/symptoms", methods=["GET"])
    def list_symptoms():
        symptoms = [
            {"id": key, "name": symptom.name, "description": symptom.description}
            for key, symptom in SYMPTOMS_DB.items()
        ]
        return jsonify({"symptoms": symptoms})

    @app.route("/api/conditions", methods=["GET"])
    def list_conditions():
        conditions = [
            {"id": key, "name": condition.name, "description": condition.description}
            for key, condition in CONDITIONS_DB.items()
        ]
        return jsonify({"conditions": conditions})

    @app.route("/api/wellness", methods=["GET"])
    def list_wellness_tips():
        tips = [
            {"category": tip.category, "title": tip.title, "tips": tip.tips}
            for tip in WELLNESS_TIPS
        ]
        return jsonify({"wellness_tips": tips})

    @app.route("/api/session/reset", methods=["POST"])
    def reset_session():
        data = request.get_json(silent=True) or {}
        session_id = data.get("session_id", "default")

        if session_id in agents:
            agents[session_id].reset()
            del agents[session_id]

        return jsonify({"status": "reset", "session_id": session_id})

    return app


app = create_app()


if __name__ == "__main__":
    # Dev server only. For production, run gunicorn (see Dockerfile / README).
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
