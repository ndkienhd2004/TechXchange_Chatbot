# Step 4 Implementation Notes

Step 4 upgrades the chatbot from in-memory state to a database-backed runtime with split storage.

## What is implemented
- App database via `APP_DATABASE_URL` for assistant conversations and messages
- Dedicated chatbot database via `CHATBOT_DATABASE_URL` for knowledge documents, chunks, embeddings, and sync state
- Automatic chatbot database bootstrap for PostgreSQL (`techxchange_chatbot` by default)
- Automatic scheduler that syncs source product data from `SOURCE_DATABASE_URL`
- PG-ready retrieval path:
  - uses `embedding_json` everywhere
  - upgrades to `embedding_vector` + DB vector search when the PostgreSQL `vector` extension is available
  - falls back to Python cosine similarity plus DB keyword search when `vector` is unavailable
- Legacy seed knowledge documents are purged from startup flow
- Runtime mock embedding and generation paths have been removed

## Runtime behavior
- All database connection strings are configured directly in the chatbot repo `.env`
- On startup, the API initializes app-side assistant tables.
- On startup, the API initializes chatbot knowledge tables.
- On startup, the API auto-creates the dedicated chatbot PostgreSQL database if needed.
- On startup, the API removes any old seed knowledge documents if they still exist.
- If `SYNC_ENABLED=true` and `SOURCE_DATABASE_URL` is set, the service runs a sync on startup and then repeats every `SYNC_INTERVAL_SECONDS`.
- Chat queries now read/write conversations in the app database.
- Chat queries now read knowledge chunks from the chatbot database instead of in-memory state.

## Current source sync scope
- Syncs active products from the main backend database.
- Pulls joined data from:
  - `products`
  - `product_catalog`
  - `brand`
  - `product_categories`
  - `stores`

## Current limitations
1. The local machine must have the PostgreSQL `vector` extension installed before DB-side vector search can turn on.
2. Without `vector`, retrieval still uses stored embeddings, but cosine scoring runs in Python rather than PostgreSQL.
3. Source sync currently covers active product data only.

## Next improvements
1. Install and enable the PostgreSQL `vector` extension on the chatbot database host.
2. Add incremental sync cursors instead of full active-product scan.
3. Add sync coverage for policies, FAQs, and other business documents from source systems.
4. Add structured logging and metrics for sync duration and failure rates.
