# TechXchange Chatbot

Python implementation of the Step 3 assistant skeleton for the TechXchange chatbot.

## What is included
- FastAPI backend for assistant chat
- Advanced RAG pipeline structure: rewrite -> embed -> retrieve -> rerank -> generate
- Gemini-based embedding and generation providers
- Hybrid storage architecture
- App database stores assistant conversations and messages
- Chatbot database stores knowledge chunks, embeddings, and sync state
- Automatic source sync scheduler for pulling product data from the main backend database

## Quick start
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python3 main.py
```

Server default:
- `http://localhost:8000`

Database defaults:
- Configure all database URLs directly in this repo's `.env`
- `APP_DATABASE_URL` points to the main app database that stores assistant conversations and messages
- `CHATBOT_DATABASE_URL` points to the dedicated chatbot database that stores KB documents, chunks, embeddings, and sync state
- `SOURCE_DATABASE_URL` points to the source database used by the sync job
- `AUTO_BOOTSTRAP_CHATBOT_DATABASE=true` lets the service create the chatbot database automatically if it does not exist yet
- PostgreSQL URLs can be provided as `postgresql://...` or `postgres://...`; the app normalizes them to the psycopg driver automatically
- If PostgreSQL is not available, local SQLite still works as a fallback by setting explicit SQLite URLs

Vector retrieval:
- The chatbot always stores `embedding_json` for compatibility
- If the chatbot database is PostgreSQL and the `vector` extension is available, startup adds an `embedding_vector` column and enables DB-side vector search
- If `vector` is unavailable, retrieval falls back to Python cosine similarity over stored embeddings plus DB keyword search

Gemini requirements:
- `GEMINI_API_KEY` is required for both embedding and generation
- `ENABLE_GEMINI_CHAT=true` and `ENABLE_GEMINI_EMBED=true` must be set for chat requests to run
- If Gemini is not configured correctly, chat and ingest endpoints return a provider/configuration error instead of using synthetic responses

## API endpoints
- `GET /health`
- `GET /api/assistant/health`
- `POST /api/assistant/chat`
- `GET /api/assistant/conversations`
- `GET /api/assistant/messages/{conversation_id}`
- `POST /api/assistant/ingest`
- `POST /api/assistant/reindex`

## Local auth
- By default, `ASSISTANT_REQUIRE_AUTH=false`, so requests use a local test user automatically.
- To test role-aware routes, pass:
  - `Authorization: Bearer demo-user-1`
  - `x-user-role: admin`

## Example request
```bash
curl -X POST http://localhost:8000/api/assistant/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "message": "Build PC tam 20 trieu de choi game AAA",
    "locale": "vi-VN"
  }'
```

## Notes
- Step 4 now stores assistant conversations/messages in the app database and stores knowledge/embeddings in the chatbot database.
- The scheduler can sync active product data from the main backend database on startup and on a fixed interval.
- Legacy seed knowledge data has been removed from the codebase and startup flow.
- The chatbot now answers only from real data already stored in the chatbot database.
