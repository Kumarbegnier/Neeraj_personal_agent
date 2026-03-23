# Neeraj AI OS

Neeraj AI OS is a research-grade personal AI agent platform built as a stateful cognitive runtime, not a chatbot. The system treats the LLM as one component inside a larger architecture that includes planning, memory, tool use, verification, reflection, safety gating, audit logging, and modular specialist orchestration.

## Project Overview

The platform is designed around a closed-loop agent runtime:

`observe -> control -> plan -> route -> agent decide -> execute -> verify -> reflect -> update state`

Key properties:

- Single evolving `AgentState` for runtime control.
- Specialized agents for communication, coding, research, browser, task, and file workflows.
- Shared tool registry with structured tool calls and structured results.
- Working memory in runtime state, episodic memory in MongoDB, semantic memory through a FAISS-compatible abstraction.
- Verification and reflection are binding, not decorative.
- Audit logging records every agent step and tool call.
- FastAPI exposes health, planning, chat, and execution APIs.

## Architecture Overview

The platform is organized into the following layers:

1. Interface layer
   FastAPI endpoints under `src/api`.
2. Backend runtime facade
   `src/runtime` re-exports the active orchestration contracts and workflow metadata used by the backend and frontend.
3. Session and context layer
   Gateway headers, request normalization, session permissions, and context construction.
4. Orchestrator
   The closed-loop controller in `agent_runtime/orchestrator.py`.
5. Planner
   Builds ordered, typed plans with verification focus and failure modes.
6. Executor / router
   Selects the right specialist and executes via the shared registry.
7. Specialized agents
   Communication, Coding, Research, Browser, Task, and File agents.
8. Tool abstraction layer
   Shared tool registry with starter tools and structured outputs.
9. Memory system
   Working memory, Mongo-backed episodic memory, semantic lookup, and a dedicated `MemoryManager`.
10. Reflection / verification layer
   Deterministic validation and one-step retry / replan signaling.
11. Safety / permissions layer
   Risk classification, approval gates, policy checks, and backend audit-log access.
12. Audit / observability layer
   Trace events, state transitions, tool-call logging, and typed audit event retrieval.

## Folder Structure

```text
main.py
requirements.txt
.env.example
README.md
ingest.py

src/
  api/
  core/
  graph/
  agents/
  tools/
  memory/
  runtime/
  schemas/
  services/
  safety/
  utils/

agent_runtime/
  orchestrator.py
  models.py
  context_hub.py
  specialists/
  control.py
  planner.py
  router.py
  router_executor.py
  execution.py
  agents.py
  tools.py
  verification.py
  reflection.py
  safety.py
  responder.py
  memory.py
  response_helpers.py
  runtime_utils.py

frontend/
  bootstrap.py
  config.py
  controller.py
  view_models.py
  services/
    api_client.py
  components/
    chat_view.py
    sidebar.py
    status_panels.py
  utils/
    state.py

pages/
  1_Chat.py
  2_Agents.py
  3_Memory.py
  4_Logs.py
  5_Settings.py

app.py
.streamlit/config.toml
```

## Core Runtime Design

The live agent runtime lives in `agent_runtime/` and uses explicit causal closure:

- `S_{t+1} = F(S_t, O_t)`
- each iteration records a `StateTransition`
- verification can force retry or replanning
- reflection mutates `adaptive_constraints`, `blocked_tools`, and `route_bias`
- memory retrieval affects planning and routing
- final response generation happens only after loop termination

## Specialized Agents

The platform includes:

- `CommunicationAgent`
- `CodingAgent`
- `ResearchAgent`
- `BrowserAgent`
- `TaskAgent`
- `FileAgent`
- `GeneralAgent`

Each specialist exposes a consistent interface and uses the shared tool layer rather than direct ad hoc calls.

The implementation now keeps three layers aligned:

- `src/agents/catalog.py` defines the typed agent catalog used by the backend and frontend.
- `agent_runtime/specialists/` contains the live specialist behavior used by the orchestrator.
- `agent_runtime/agents.py` keeps the public import path stable as a compatibility facade.

## Shared Tool Registry

The shared registry includes:

- memory surfaces such as `session_history`, `semantic_memory`, `vector_memory`, `working_memory`, and `save_memory`
- planning and safety surfaces such as `capability_map`, `plan_analyzer`, `verification_harness`, and `risk_monitor`
- integration surfaces such as `github_adapter`, `calendar_adapter`, `document_adapter`, and `browser_adapter`
- `send_email_draft`
- `search_web`
- `browser_search`
- `load_recent_memory`
- `summarize_file`
- `generate_code`
- `open_page`
- `extract_page_text`
- `create_task_record`

These tools return structured results and fail gracefully when external dependencies or credentials are absent.

The typed tool catalog now lives in `src/tools/catalog.py`, which is shared by the runtime, API layer, and Streamlit frontend.

## API Endpoints

- `GET /health`
- `GET /agents`
- `GET /tools`
- `GET /audit/logs`
- `POST /chat`
- `POST /plan`
- `POST /execute`

Compatibility routes are also preserved:

- `GET /status`
- `GET /architecture`
- `GET /sessions/{user_id}/{session_id}`
- `POST /interactions`

The Streamlit frontend uses the backend API to provide:

- chat workspace
- planner / task breakdown
- memory / context review
- agent status / execution review
- logs / audit review
- settings and session controls

The backend now also exposes typed runtime metadata for:

- specialist catalog inspection
- shared tool registry inspection
- recent audit event retrieval

## Local Setup

### 1. Create a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Copy environment variables

```powershell
Copy-Item .env.example .env
```

### 4. Install Playwright browser binaries

```powershell
playwright install chromium
```

### 5. Run the API

```powershell
uvicorn main:app --reload
```

The app is designed to run locally on Windows with:

```powershell
uvicorn main:app --reload
```

### 6. Run the Streamlit frontend

Start the backend first, then launch the research console:

```powershell
streamlit run app.py
```

By default the frontend targets `http://127.0.0.1:8000`. You can override that with:

```powershell
$env:NEERAJ_API_URL="http://127.0.0.1:8000"
streamlit run app.py
```

## Environment Variables

Important settings:

- `OPENAI_API_KEY`
- `OPENAI_CHAT_MODEL`
- `OPENAI_RESPONSES_MODEL`
- `OPENAI_EMBEDDING_MODEL`
- `USE_OPENAI_AGENTS_SDK`
- `MONGODB_URI`
- `MONGODB_DB_NAME`
- `MONGODB_EPISODIC_COLLECTION`
- `MONGODB_TASK_COLLECTION`
- `PLAYWRIGHT_BROWSER`
- `PLAYWRIGHT_HEADLESS`
- `AUDIT_LOG_FILE`
- `NEERAJ_API_URL`
- `NEERAJ_API_TIMEOUT_SECONDS`

If `OPENAI_API_KEY` or the Agents SDK is unavailable, the platform still runs in deterministic local mode.

## MongoDB Setup

For local development, a local MongoDB instance is enough:

```powershell
docker run -d --name neeraj-mongo -p 27017:27017 mongo:7
```

Then set:

```env
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB_NAME=neeraj_ai_os
```

MongoDB is used for:

- durable episodic memory
- task outcome logs
- future expansion to richer long-term memory

## Playwright Setup

Install Playwright and browser binaries:

```powershell
pip install playwright
playwright install chromium
```

The current code exposes browser-first tool abstractions and safe placeholders. This keeps the platform runnable even before live browser automation is fully wired.

## Example Requests

### Health

```powershell
curl http://127.0.0.1:8000/health
```

### Plan

```powershell
curl -X POST http://127.0.0.1:8000/plan ^
  -H "Content-Type: application/json" ^
  -d "{\"user_id\":\"demo\",\"session_id\":\"plan-1\",\"channel\":\"text\",\"message\":\"Design a modular personal AI agent platform.\"}"
```

### Chat

```powershell
curl -X POST http://127.0.0.1:8000/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"user_id\":\"demo\",\"session_id\":\"chat-1\",\"channel\":\"text\",\"message\":\"Build a research-grade coding agent architecture.\"}"
```

### Execute

```powershell
curl -X POST http://127.0.0.1:8000/execute ^
  -H "Content-Type: application/json" ^
  -d "{\"user_id\":\"demo\",\"session_id\":\"exec-1\",\"channel\":\"text\",\"message\":\"Create a plan and execute a browser-assisted research workflow.\"}"
```

## OpenAI Agents SDK Integration

The project includes a lightweight integration layer in `src/services/llm_service.py` that follows OpenAI Agents SDK idioms:

- build SDK agents from structured specialist definitions
- support OpenAI Responses-compatible models
- degrade safely when SDK dependencies or credentials are missing

This keeps the architecture extensible without making the first version depend on always-online model execution.

## Observability

The platform records:

- per-stage trace events
- per-iteration state transitions
- tool-call audit events
- durable interaction logs
- typed audit events exposed through `GET /audit/logs`

This supports debugging, evaluation, and future production tracing integrations.

## Roadmap

- Replace stubbed search and browser tools with live OpenAI / Playwright-backed integrations.
- Add durable semantic vector persistence behind the FAISS-compatible abstraction.
- Add real OpenAI Agents SDK function tools and handoffs for specialist delegation.
- Expand safety guardrails and human approval workflows.
- Add evaluation harnesses and regression tests for multi-step agent workflows.

## First Recommended Next Step

Implement live external tool adapters for `search_web`, `browser_search`, `open_page`, and `extract_page_text`, then connect them to real OpenAI Agents SDK function tools and Playwright browser sessions. That will convert the current research-grade architecture from a safe local platform into a genuinely operational personal AI agent system.
