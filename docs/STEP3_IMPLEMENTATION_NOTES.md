# Step 3 Implementation Notes

Python version of Step 3 now includes:
- FastAPI backend and assistant routes
- Initial RAG runtime path and assistant API skeleton
- Admin ingest and reindex endpoints
- Gemini integration points for generation and embedding
- In-memory conversation state for local testing

Next step candidates:
1. Replace in-memory conversation store with dedicated chatbot database tables.
2. Sync knowledge from the main backend source automatically on a schedule.
3. Plug frontend assistant mode to `/api/assistant/chat`.
4. Add request logging and metrics export.
