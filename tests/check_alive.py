#!/usr/bin/env python
"""SUQL liveness probe — one query, one verdict.

The whole point of this script: prove that a SUQL stack — Postgres,
embedding server, free-text server, LLM proxy — is end-to-end functional.
It is the script GitHub Actions runs at the end of the pipeline to decide
whether the build is green.

Every connection target is an env var, so the same script runs unchanged
locally and in CI.

Exit codes:
    0   SUQL returned >= 1 row
    1   SUQL raised — connection, server, LLM, or compiler error
    2   SUQL ran cleanly but returned 0 rows (alive but suspicious)
    3   preflight failed — a TCP target was unreachable or creds missing

Usage::

    python tests/check_alive.py                  # use env / defaults
    python tests/check_alive.py --print-env      # show resolved config, exit
    python tests/check_alive.py --no-preflight   # skip TCP probe, fail inside SUQL instead
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Bootstrap: load .env if present, ensure suql imports from this repo's src/
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent

try:
    from dotenv import load_dotenv

    env_path = _REPO_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # In CI, env vars come from the workflow directly

_SRC = _REPO_ROOT / "src"
if _SRC.exists():
    sys.path.insert(0, str(_SRC))


# Defaults assume the workflow's service containers + the example fixtures
# schema. Override any of these from the calling environment for a different
# database, table layout, or LLM proxy.
DEFAULTS = {
    "PGHOST":            "127.0.0.1",
    "PGPORT":            "5432",
    "PGDATABASE":        "acled",
    "PGUSER":            "oval",
    "PGPASSWORD":        "oval",
    "SUQL_EMBED_URL":    "http://127.0.0.1:8505",
    "SUQL_FREETEXT_URL": "http://127.0.0.1:8500",
    "SUQL_SELECT_USER":  "select_user",
    "SUQL_SELECT_PSWD":  "select_user",
    "SUQL_CREATE_USER":  "creator_role",
    "SUQL_CREATE_PSWD":  "creator_role",
    "SUQL_TABLE":        "events",
    "SUQL_ID_COL":       "event_id_cnty",
    "SUQL_TIMEOUT_MS":   "120000",
}

# Override with SUQL_QUERY env var. The default below pairs with the example
# schema in tests/fixtures/schema.sql; any query that exercises one answer()
# clause and returns >= 1 row from your data works.
DEFAULT_QUERY = """\
SELECT event_id_cnty, country, event_type, notes
FROM events
WHERE notes IS NOT NULL
  AND answer(notes, 'Does this describe a conflict, protest, or violent event?') = 'yes'
LIMIT 3;
"""


def _env(key: str) -> str:
    return os.environ.get(key, DEFAULTS.get(key, ""))


def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _preflight() -> list[str]:
    problems: list[str] = []
    if not _port_open(_env("PGHOST"), int(_env("PGPORT"))):
        problems.append(f"Postgres unreachable at {_env('PGHOST')}:{_env('PGPORT')}")
    for label, key in [("embedding", "SUQL_EMBED_URL"), ("free-text", "SUQL_FREETEXT_URL")]:
        u = urlparse(_env(key))
        port = u.port or (443 if u.scheme == "https" else 80)
        if not _port_open(u.hostname or "127.0.0.1", port):
            problems.append(f"SUQL {label} server unreachable at {_env(key)}")
    if not os.environ.get("OPENAI_API_KEY") or not os.environ.get("OPENAI_API_BASE"):
        problems.append("OPENAI_API_KEY / OPENAI_API_BASE not set")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="SUQL liveness probe")
    parser.add_argument("--print-env", action="store_true",
                        help="Show resolved config and exit.")
    parser.add_argument("--no-preflight", action="store_true",
                        help="Skip TCP / cred preflight; let SUQL fail instead.")
    args = parser.parse_args()

    sql = os.environ.get("SUQL_QUERY", DEFAULT_QUERY).strip()
    config = {k: _env(k) for k in DEFAULTS}
    config["OPENAI_API_BASE"] = os.environ.get("OPENAI_API_BASE", "<unset>")
    config["OPENAI_API_KEY"]  = "<set>" if os.environ.get("OPENAI_API_KEY") else "<unset>"

    print("=== SUQL liveness probe ===")
    for k, v in config.items():
        print(f"  {k:18s} {v}")
    print()
    print("Query:")
    for line in sql.splitlines():
        print(f"  {line}")
    print()

    if args.print_env:
        return 0

    if not args.no_preflight:
        problems = _preflight()
        if problems:
            print("✗ Preflight failed:")
            for p in problems:
                print(f"    - {p}")
            return 3
        print("✓ Preflight: all ports open, API creds set")
        print()

    try:
        from suql import suql_execute
    except ImportError as e:
        print(f"✗ Cannot import suql: {e}")
        return 1

    t0 = time.time()
    try:
        rows, cols, cache = suql_execute(
            sql,
            table_w_ids={_env("SUQL_TABLE"): _env("SUQL_ID_COL")},
            database=_env("PGDATABASE"),
            embedding_server_address=_env("SUQL_EMBED_URL"),
            free_text_server_address=_env("SUQL_FREETEXT_URL"),
            host=_env("PGHOST"),
            port=int(_env("PGPORT")),
            select_username=_env("SUQL_SELECT_USER"),
            select_userpswd=_env("SUQL_SELECT_PSWD"),
            create_username=_env("SUQL_CREATE_USER"),
            create_userpswd=_env("SUQL_CREATE_PSWD"),
            statement_timeout=int(_env("SUQL_TIMEOUT_MS")),
            api_base=os.environ.get("OPENAI_API_BASE"),
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
    except Exception as e:
        elapsed = time.time() - t0
        print(f"✗ suql_execute raised after {elapsed:.1f}s")
        print(f"    {type(e).__name__}: {e}")
        return 1

    elapsed = time.time() - t0
    stats = (cache or {}).get("_stats", {})
    print(f"✓ suql_execute returned in {elapsed:.1f}s")
    print(f"    rows:  {len(rows)}")
    print(f"    cols:  {cols}")
    print(f"    cost:  ${stats.get('cost', 0):.4f}")
    print(f"    calls: {stats.get('calls', 0)}")
    if rows:
        first = rows[0]
        preview = [(str(v)[:80] + "…") if len(str(v)) > 80 else str(v) for v in first]
        print(f"    row[0]: {preview}")
        return 0

    print()
    print("⚠ Query ran cleanly but returned zero rows.")
    print("  SUQL is alive but either (a) the LLM judged every candidate as 'no',")
    print("  (b) the WHERE clause filtered everything out, or (c) the embedding")
    print("  server returned no neighbours. Inspect cache for details.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
