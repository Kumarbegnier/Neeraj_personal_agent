# Flash Space AI Agent Backend

FastAPI backend for a multi-tenant AI assistant with:
- JWT auth
- session-based memory (SQLite)
- RAG retrieval (Pinecone)
- LLM + embeddings via proxy

## Tech Stack

- Python
- FastAPI
- LangChain
- SQLAlchemy (SQLite)
- Pinecone

## Project Structure

```text
app/
  main.py              # FastAPI entrypoint
  auth.py              # JWT auth + AuthContext
  chains.py            # Prompt + LLM chain with memory
  llm.py               # LLM provider selection (proxy/offline fallback)
  prompts.py           # Prompt loader and template
  memory.py            # LangChain chat history backed by SQLite
  session_store.py     # DB CRUD for sessions/messages
  vectorstore.py       # Pinecone retriever + embeddings + upsert
  ingestion/           # JSON source/chunking/Mongo helpers
prompts/
  system.txt
  user.txt
scripts/
  cli_chat.py          # Terminal chat client
docs/
  ARCHITECTURE.md
```

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Create `.env` in project root.

3. Run API:

```bash
uvicorn app.main:app --reload
```

4. Health check:

```bash
curl http://127.0.0.1:8000/health
```

## Core Environment Variables

### Auth

- `JWT_SECRET` (required unless `DEV_AUTH_BYPASS=true`)
- `JWT_ALGORITHM` (default: `HS256`)
- `JWT_ISSUER` (optional)
- `JWT_AUDIENCE` (optional)
- `JWT_USER_CLAIM` (default: `sub`)
- `JWT_TENANT_CLAIM` (default: `tenant_id`)
- `JWT_ROLE_CLAIM` (default: `role`)

### Dev auth bypass (local only)

- `DEV_AUTH_BYPASS=true`
- `DEV_USER_ID=dev_user`
- `DEV_TENANT_ID=dev_tenant`
- `DEV_ROLE=admin`

### LLM / Embeddings

- `PROXY_URL` (recommended)
- `OPENAI_MODEL` (default: `gpt-4o-mini`)
- `OPENAI_EMBEDDING_MODEL` (default: `text-embedding-3-small`)
- `OPENAI_EMBEDDING_DIMENSIONS` (optional int)

### Vector DB

- `PINECONE_API_KEY`
- `PINECONE_INDEX_NAME`
- `PINECONE_TEXT_KEY` (default: `text`)

### Database

- `DATABASE_URL` (default: `sqlite:///./chat.db`)

### Optional Mongo ingestion

- `MONGODB_URI`
- `MONGODB_DB` (default: `automation1`)
- `MONGODB_COLLECTION` (default: `spaces`)

## API

### `POST /chat`

Request:

```json
{
  "query": "What is virtual office pricing in Delhi?",
  "conversation_id": "default",
  "session_id": "optional-session-id"
}
```

Response:

```json
{
  "reply": "....",
  "session_id": "tenant:user:default"
}
```

### `GET /health`

Response:

```json
{
  "status": "ok"
}
```

## CLI Chat (Optional)

```bash
python scripts/cli_chat.py --base-url http://127.0.0.1:8000
```

Supports commands:
- `:token`
- `:conv <id>`
- `:session`
- `:session <id>`
- `exit`

## Architecture

Detailed architecture and flow diagrams:
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
