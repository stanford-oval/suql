#!/usr/bin/env python
"""Start the SUQL embedding server pointed at the ACLED ``events`` table.

SUQL's bundled ``python -m suql.faiss_embedding`` is hard-coded for the
restaurants demo (see ``faiss_embedding.py``'s ``__main__``). This script
is the ACLED equivalent: register ``events.notes`` with a
``MultipleEmbeddingStore`` and start the HTTP server.

Reads from env (with sensible defaults that match the CI workflow):
    PGDATABASE          (default: acled)
    PGUSER              (default: oval)
    PGPASSWORD          (default: oval)
    SUQL_EMBED_PORT     (default: 8505)
    SUQL_TABLE          (default: events)
    SUQL_ID_COL         (default: event_id_cnty)
    SUQL_TEXT_COL       (default: notes)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make sure we import the checked-out suql, not a stray pip-installed copy.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if _SRC.exists():
    sys.path.insert(0, str(_SRC))

from suql.faiss_embedding import MultipleEmbeddingStore


def main() -> None:
    store = MultipleEmbeddingStore()
    store.add(
        table_name=os.environ.get("SUQL_TABLE", "events"),
        primary_key_field_name=os.environ.get("SUQL_ID_COL", "event_id_cnty"),
        free_text_field_name=os.environ.get("SUQL_TEXT_COL", "notes"),
        db_name=os.environ.get("PGDATABASE", "acled"),
        user=os.environ.get("PGUSER", "oval"),
        password=os.environ.get("PGPASSWORD", "oval"),
    )
    store.start_embedding_server(
        host="127.0.0.1",
        port=int(os.environ.get("SUQL_EMBED_PORT", "8505")),
    )


if __name__ == "__main__":
    main()
