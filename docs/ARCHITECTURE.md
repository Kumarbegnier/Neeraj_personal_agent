# AI Agent Backend Architecture

## 1) High-Level Overview

```mermaid
flowchart LR
    C[Client / Frontend / CLI] -->|POST /chat + Bearer JWT| API[FastAPI app.main]
    API --> AUTH[auth.get_auth_context]
    AUTH --> API
    API --> RET[vectorstore.retrieve_with_scores]
    RET --> PC[(Pinecone Index)]
    RET --> EMB[OpenAI Embeddings via Proxy]
    API --> CHAIN[chains.build_chain]
    CHAIN --> PROMPT[prompts: system.txt + user.txt]
    CHAIN --> LLM[ChatOpenAI via Proxy]
    CHAIN --> MEM[memory.SQLiteChatMessageHistory]
    MEM --> DB[(SQLite chat.db)]
    API --> RES[reply + session_id]
    RES --> C
```

## 2) Core Runtime Components

- API Layer: `app/main.py`
  - Endpoints: `/chat`, `/health`
  - Builds session id (`tenant:user:conversation`) if `session_id` missing
  - Retrieves RAG context and invokes LangChain pipeline

- Auth Layer: `app/auth.py`
  - Validates Bearer JWT (`JWT_SECRET`, issuer/audience optional)
  - Extracts `user_id`, `tenant_id`, `role`
  - Creates namespace: `tenant_id:role`
  - Supports local dev bypass via `DEV_AUTH_BYPASS=true`

- Chain Layer: `app/chains.py`, `app/prompts.py`, `app/llm.py`
  - Prompt template: system + history + user question
  - LLM source: Proxy-backed OpenAI (`PROXY_URL`)
  - Fallback: offline mock response when LLM provider unavailable

- Memory Layer: `app/memory.py`, `app/session_store.py`, `app/models.py`
  - Conversation history persisted in SQLite (`chat.db`)
  - Tables:
    - `chat_sessions`
    - `chat_messages`
  - LangChain `RunnableWithMessageHistory` reads/writes chat history automatically

- Retrieval Layer (RAG): `app/vectorstore.py`
  - Vector DB: Pinecone
  - Embeddings: OpenAI embeddings through proxy
  - Retrieval method: `similarity_search_with_score(k=4)`
  - Injects joined document text into prompt as `context`

## 3) /chat Request Flow

```mermaid
sequenceDiagram
    participant U as User/Client
    participant A as FastAPI /chat
    participant AU as Auth
    participant S as Session Store (SQLite)
    participant V as Vectorstore (Pinecone)
    participant C as LangChain Chain
    participant L as LLM Proxy

    U->>A: POST /chat {query, conversation_id?, session_id?}
    A->>AU: Validate Bearer JWT
    AU-->>A: AuthContext(user_id, tenant_id, role, namespace)
    A->>S: ensure_session(session_id)
    A->>V: retrieve_with_scores(namespace, query, k=4)
    V-->>A: docs + scores
    A->>C: invoke({question, context}, session_id)
    C->>S: load history messages
    C->>L: send prompt(system + history + question + context)
    L-->>C: AI response
    C->>S: save user/ai messages
    C-->>A: response.content
    A-->>U: {reply, session_id}
```

## 4) Ingestion/Data Preparation Flow

```mermaid
flowchart LR
    J[JSON Source] --> JS[ingestion.json_source.load_spaces_from_json]
    JS --> TXT[records_to_texts]
    TXT --> CH[ingestion.chunking.chunk_texts]
    CH --> UP[vectorstore.upsert_texts]
    UP --> PC[(Pinecone namespace)]
    JS --> MONGO[ingestion.mongo.upsert_space_records optional]
    MONGO --> MDB[(MongoDB)]
```

## 5) Configuration Dependencies

- LLM/Embeddings
  - `PROXY_URL`
  - `OPENAI_MODEL` (optional)
  - `OPENAI_EMBEDDING_MODEL` (optional)
  - `OPENAI_EMBEDDING_DIMENSIONS` (optional)

- Pinecone
  - `PINECONE_API_KEY`
  - `PINECONE_INDEX_NAME`
  - `PINECONE_TEXT_KEY` (optional)

- Auth
  - `JWT_SECRET`
  - `JWT_ALGORITHM` (default `HS256`)
  - `JWT_ISSUER`, `JWT_AUDIENCE` (optional)
  - Claim mapping env vars (`JWT_USER_CLAIM`, etc.)

- Database
  - `DATABASE_URL` (default `sqlite:///./chat.db`)

- Optional Mongo ingestion
  - `MONGODB_URI`
  - `MONGODB_DB`
  - `MONGODB_COLLECTION`

## 6) Design Characteristics

- Multi-tenant separation via `namespace = tenant_id:role` at retrieval layer
- Persistent conversational memory keyed by `session_id`
- Graceful degradation:
  - Retrieval failure => empty context, chat still runs
  - LLM provider missing => offline fallback response
