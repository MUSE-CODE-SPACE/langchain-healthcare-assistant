# AI Healthcare Assistant

LangChain-powered health information assistant that provides general health information and wellness tips.

**IMPORTANT DISCLAIMER**: This application provides general health information for educational purposes only. It is NOT a substitute for professional medical advice, diagnosis, or treatment.

## Live Demo

[**View Demo**](https://yoon-k.github.io/langchain-healthcare-assistant/)

## Features

- **Symptom Information**: Get general information about common symptoms
- **Condition Overviews**: Learn about health conditions and their management
- **Wellness Tips**: Nutrition, exercise, sleep, and mental health guidance
- **Health Calculators**: BMI calculator, water intake recommendations
- **Risk Assessment**: General lifestyle health assessment
- **Emergency Detection**: Recognizes emergency keywords and advises calling 911

## Architecture

```
langchain-healthcare-assistant/
├── app/
│   ├── agents/
│   │   └── health_agent.py       # Main health assistant agent
│   ├── tools/
│   │   └── health_tools.py       # LangChain health tools
│   ├── data/
│   │   └── health_knowledge.py   # Health information database
│   └── api.py                    # Flask API endpoints
├── static/
│   ├── css/style.css
│   └── js/app.js
├── templates/
│   └── index.html
└── requirements.txt
```

## LangChain Components

### Custom Tools
- `SymptomCheckerTool`: Get information about symptoms
- `ConditionInfoTool`: Learn about health conditions
- `WellnessTipsTool`: Get wellness and lifestyle tips
- `BMICalculatorTool`: Calculate Body Mass Index
- `WaterIntakeTool`: Calculate recommended water intake
- `HealthRiskAssessmentTool`: General lifestyle assessment
- `EmergencyGuidanceTool`: Emergency situation guidance

### Agent Architecture
```python
from app.agents.health_agent import create_health_agent

# Create agent
agent = create_health_agent()

# Chat with context awareness
response = agent.chat("I have a headache")
response = agent.chat("What about nutrition tips?")
response = agent.chat("Calculate my BMI for 70kg and 175cm")
```

### Health Knowledge Base
```python
from app.data.health_knowledge import SYMPTOMS_DB, CONDITIONS_DB, WELLNESS_TIPS

# Access symptom information
headache = SYMPTOMS_DB["headache"]
print(headache.common_causes)
print(headache.home_remedies)

# Access condition information
cold = CONDITIONS_DB["common_cold"]
print(cold.prevention)

# Get wellness tips
for tip in WELLNESS_TIPS:
    print(f"{tip.category}: {tip.title}")
```

## Installation

```bash
# Clone repository
git clone https://github.com/yoon-k/langchain-healthcare-assistant.git
cd langchain-healthcare-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python -m app.api
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/chat` | POST | Main chat endpoint |
| `/api/health` | GET | Health check |
| `/api/symptoms` | GET | List all symptoms |
| `/api/conditions` | GET | List all conditions |
| `/api/wellness` | GET | Get wellness tips |
| `/api/session/reset` | POST | Reset conversation |

## Available Health Information

### Symptoms
- Headache
- Fatigue
- Sore Throat
- Back Pain
- Stomach Ache
- Cough

### Conditions
- Common Cold
- Influenza (Flu)
- Allergies
- High Blood Pressure
- Type 2 Diabetes

### Wellness Categories
- Nutrition
- Exercise
- Sleep
- Mental Health
- Preventive Care

## Tech Stack

- **LangChain**: Agent framework and tool orchestration
- **Flask**: Backend API server
- **Python 3.9+**: Core language
- **Pydantic**: Data validation and tool input schemas

## Safety Features

1. **Disclaimers**: All medical information includes disclaimers
2. **Emergency Detection**: Recognizes emergency keywords and advises calling 911
3. **Professional Referral**: Always recommends consulting healthcare professionals
4. **No Diagnosis**: Never provides diagnoses or treatment recommendations

## Contributing

Contributions are welcome! Please ensure any health information added is accurate and includes appropriate disclaimers.

## License

MIT License - This is an educational project demonstrating LangChain capabilities.

## Disclaimer

This application is for educational and demonstration purposes only. The health information provided is general in nature and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
