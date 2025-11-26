"""
Flask API for Healthcare Assistant
"""

import os
import json
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from datetime import datetime

from app.agents.health_agent import HealthAssistantAgent, create_health_agent
from app.data.health_knowledge import SYMPTOMS_DB, CONDITIONS_DB, WELLNESS_TIPS


def create_app():
    app = Flask(__name__,
                static_folder='../static',
                template_folder='../templates')
    CORS(app)

    agents = {}

    def get_agent(session_id: str) -> HealthAssistantAgent:
        if session_id not in agents:
            agents[session_id] = create_health_agent()
        return agents[session_id]

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/static/<path:filename>')
    def serve_static(filename):
        return send_from_directory(app.static_folder, filename)

    @app.route('/api/health', methods=['GET'])
    def health_check():
        return jsonify({
            'status': 'healthy',
            'service': 'healthcare-assistant',
            'timestamp': datetime.utcnow().isoformat()
        })

    @app.route('/api/chat', methods=['POST'])
    def chat():
        data = request.json
        session_id = data.get('session_id', 'default')
        message = data.get('message', '')

        if not message:
            return jsonify({'error': 'Message is required'}), 400

        try:
            agent = get_agent(session_id)
            response = agent.chat(message)

            return jsonify({
                'response': response,
                'session_id': session_id
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/symptoms', methods=['GET'])
    def list_symptoms():
        symptoms = []
        for key, symptom in SYMPTOMS_DB.items():
            symptoms.append({
                'id': key,
                'name': symptom.name,
                'description': symptom.description
            })
        return jsonify({'symptoms': symptoms})

    @app.route('/api/conditions', methods=['GET'])
    def list_conditions():
        conditions = []
        for key, condition in CONDITIONS_DB.items():
            conditions.append({
                'id': key,
                'name': condition.name,
                'description': condition.description
            })
        return jsonify({'conditions': conditions})

    @app.route('/api/wellness', methods=['GET'])
    def list_wellness_tips():
        tips = []
        for tip in WELLNESS_TIPS:
            tips.append({
                'category': tip.category,
                'title': tip.title,
                'tips': tip.tips
            })
        return jsonify({'wellness_tips': tips})

    @app.route('/api/session/reset', methods=['POST'])
    def reset_session():
        data = request.json
        session_id = data.get('session_id', 'default')

        if session_id in agents:
            agents[session_id].reset()
            del agents[session_id]

        return jsonify({'status': 'reset', 'session_id': session_id})

    return app


app = create_app()


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
