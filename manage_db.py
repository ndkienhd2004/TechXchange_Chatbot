from __future__ import annotations

"""CLI helpers for DB init, sync, and manual ingestion tasks."""

import argparse
import asyncio
import json

from app.repository import app_message_repository
from app.repository import chatbot_repository
from app.source_sync import sync_from_source_database
from data_loader import load_documents_from_json


def main() -> None:
    """Parse CLI arguments and execute one management action."""

    parser = argparse.ArgumentParser(description="Manage chatbot storage and sync")
    parser.add_argument("--init-db", action="store_true", help="Create chatbot tables")
    parser.add_argument("--stats", action="store_true", help="Show chatbot DB stats")
    parser.add_argument("--sync-now", action="store_true", help="Run a source sync now")
    parser.add_argument("--ingest-json", type=str, help="Ingest documents from a JSON file")
    args = parser.parse_args()

    app_message_repository.init_database()
    chatbot_repository.init_database()

    if args.init_db:
        print(
            json.dumps(
                {
                    "initialized": True,
                    "storage": chatbot_repository.get_storage_capabilities(),
                },
                indent=2,
            )
        )
        return
    if args.stats:
        print(
            json.dumps(
                {
                    "knowledge": chatbot_repository.get_kb_stats(),
                    "storage": chatbot_repository.get_storage_capabilities(),
                    "sync": chatbot_repository.get_sync_state(),
                    "messages_db": "initialized",
                },
                indent=2,
            )
        )
        return
    if args.sync_now:
        print(json.dumps(asyncio.run(sync_from_source_database()), indent=2))
        return
    if args.ingest_json:
        docs = load_documents_from_json(args.ingest_json)
        print(
            json.dumps(
                asyncio.run(
                    chatbot_repository.upsert_documents(
                        "manual_ingest",
                        [
                            {
                                "source_key": item.get("source_key")
                                or f"manual:{index}",
                                "title": item["title"],
                                "uri": item.get("uri", ""),
                                "content": item["content"],
                                "metadata": item.get("metadata", {}),
                            }
                            for index, item in enumerate(docs, start=1)
                        ],
                        purge_missing=False,
                    )
                ),
                indent=2,
            )
        )
        return

    parser.print_help()


if __name__ == "__main__":
    main()
