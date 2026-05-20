# AI Healthcare Assistant

LangChain 0.3 powered health *information* assistant with a Flask API.

> **IMPORTANT MEDICAL DISCLAIMER**
> This application is for **educational purposes only**. It is **NOT** a
> substitute for professional medical advice, diagnosis, or treatment. It
> does NOT diagnose illness, prescribe medication, or replace clinical care.
> Always consult a qualified healthcare professional. If you think you are
> having a medical emergency, **call 911 (or your local emergency number)
> immediately**.
>
> 본 애플리케이션은 **교육 목적**의 일반 건강 정보 제공용입니다. **의학적
> 진단·치료·처방을 대체하지 않으며**, 어떠한 의료 자문도 아닙니다. 건강
> 문제가 있다면 반드시 **자격을 갖춘 의료 전문가**와 상담하세요. 응급
> 상황으로 의심되면 즉시 **119(국내) 또는 911**에 연락하십시오.

[**Live demo**](https://yoon-k.github.io/langchain-healthcare-assistant/)

---

## Features

- **Symptom information** — common causes, home care, and when to see a doctor
- **Condition overviews** — cold, flu, allergies, hypertension, type 2 diabetes
- **Wellness tips** — nutrition, exercise, sleep, mental health, prevention
- **Health calculators** — BMI, water intake, lifestyle risk score
- **Emergency detection** — pre-filter that always overrides the LLM and steers
  the user to 911 / 988 for chest pain, suicidal thoughts, trouble breathing,
  stroke / seizure / anaphylaxis / overdose keywords
- **Graceful LLM fallback** — runs without any API key (keyword router) or
  with Anthropic Claude / OpenAI GPT when keys are configured

## Quick start

```bash
git clone https://github.com/yoon-k/langchain-healthcare-assistant.git
cd langchain-healthcare-assistant

# Option A — Makefile (recommended)
make install-dev
make test
make run                          # http://localhost:5000

# Option B — manual
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
LLM_PROVIDER=none python -m app.api
```

Check it's alive:

```bash
curl http://localhost:5000/api/health
# {"status":"healthy","service":"healthcare-assistant","timestamp":"..."}

curl -X POST http://localhost:5000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"give me some nutrition tips"}'
```

## LLM modes

The agent picks a backend based on environment variables:

| `LLM_PROVIDER` | Required keys                       | Default model                 | Behaviour |
| -------------- | ----------------------------------- | ----------------------------- | --------- |
| `anthropic`    | `ANTHROPIC_API_KEY`                 | `claude-haiku-4-5-20251001`   | Tool-calling agent (LCEL) |
| `openai`       | `OPENAI_API_KEY`                    | `gpt-4o-mini`                 | Tool-calling agent (LCEL) |
| `none`         | —                                   | —                             | Deterministic keyword router (offline) |
| *(unset)*      | auto-detect from any key present    | as above                      | Falls back to `none` if no keys |

Override the model with `ANTHROPIC_MODEL` / `OPENAI_MODEL`.

**The emergency pre-filter runs in every mode, before the LLM is ever
called.** If the user message contains any of the high-risk keywords
(`chest pain`, `can't breathe`, `suicide`, `stroke`, `overdose`, …) the
response is the hard-coded emergency template that points to 911 / 988.

Copy `.env.example` to `.env` and fill in the values you need.

## Architecture

```
langchain-healthcare-assistant/
├── app/
│   ├── agents/health_agent.py      # LCEL tool-calling agent + keyword fallback
│   ├── tools/health_tools.py       # 7 BaseTool implementations (Pydantic v2)
│   ├── data/health_knowledge.py    # Dataclass-backed symptom/condition/wellness DB
│   └── api.py                      # Flask blueprint
├── tests/test_smoke.py             # pytest smoke tests (no network)
├── docs/index.html                 # Static demo page
├── Dockerfile                      # python:3.12-slim + gunicorn
├── Makefile                        # make install-dev / test / run / docker-build
├── pyproject.toml                  # PEP 621 + ruff / mypy / pytest config
└── requirements.txt
```

## API endpoints

| Endpoint              | Method | Description                                          |
| --------------------- | ------ | ---------------------------------------------------- |
| `/api/health`         | GET    | Liveness probe                                       |
| `/api/chat`           | POST   | `{ message, session_id? }` → `{ response, mode, … }` |
| `/api/symptoms`       | GET    | List of supported symptoms                           |
| `/api/conditions`     | GET    | List of supported conditions                         |
| `/api/wellness`       | GET    | Wellness tips by category                            |
| `/api/session/reset`  | POST   | `{ session_id }` — clear a session                   |

## LangChain components

- **Tools** (`langchain_core.tools.BaseTool`): `SymptomCheckerTool`,
  `ConditionInfoTool`, `WellnessTipsTool`, `BMICalculatorTool`,
  `WaterIntakeTool`, `HealthRiskAssessmentTool`, `EmergencyGuidanceTool`.
- **Agent** (LangChain 0.3): `create_tool_calling_agent` + `AgentExecutor`,
  with a `ChatPromptTemplate` system prompt that bakes in the medical
  disclaimer.
- **Memory**: a windowed `collections.deque` (no deprecated
  `ConversationBufferWindowMemory`).

## Development

```bash
make install-dev        # editable install with pytest / ruff / mypy
make lint               # ruff check .
make fmt                # ruff format + autofix
make test               # pytest
make typecheck          # mypy app
```

CI runs ruff + pytest on Python 3.12 (`.github/workflows/ci.yml`).

## Docker (production)

```bash
make docker-build
make docker-run PORT=5000
# or:
docker run --rm -p 5000:5000 \
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  langchain-healthcare-assistant
```

The image runs as a non-root user and serves the app with `gunicorn`
(`--workers 2`). The Flask dev server is *only* for local development.

## Safety features

1. **Emergency keyword pre-filter** runs before any LLM call.
2. **Disclaimer is appended automatically** when the LLM response omits it.
3. **System prompt** explicitly forbids diagnoses, prescriptions, and
   dosage recommendations.
4. **No persistent storage** of user messages beyond the per-session
   in-memory window.

## Tech stack

- LangChain 0.3 (langchain, langchain-core, langchain-community,
  langchain-anthropic, langchain-openai)
- Flask 3.1 + flask-cors 5
- Pydantic 2.10
- Python 3.12+

## License

MIT. Educational project — not for clinical use.

## 한국어 안내 (Korean)

이 프로젝트는 **건강 정보**를 제공하는 LangChain 기반 어시스턴트입니다.
의료 진단/처방이 아니며, 모든 응답은 교육 목적입니다.

- API 키 없이도 동작합니다 (`LLM_PROVIDER=none`, 키워드 라우터 모드).
- `ANTHROPIC_API_KEY` 또는 `OPENAI_API_KEY`를 설정하면 실제 LLM(도구
  호출 에이전트) 모드로 자동 전환됩니다.
- 가슴 통증, 호흡 곤란, 자살 충동, 뇌졸중, 과다 복용 등 응급 키워드는
  LLM 호출 *이전*에 차단되어 즉시 응급 안내 메시지를 반환합니다.
- 응급 상황이 의심되면 즉시 **119**(국내) 또는 911에 전화하세요. 자살
  위기는 한국에서는 **자살예방 상담전화 1393**, 미국에서는 **988**.

빠른 시작:

```bash
make install-dev
make test
make run            # http://localhost:5000
```
