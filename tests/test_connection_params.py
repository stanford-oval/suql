"""Regression tests: `host` and `port` given to `suql_execute` have to reach
every database connection the compiler opens.

A query with `answer()` in its WHERE clause is compiled by
`_analyze_SelectStmt`, which used to take no `host`/`port`, so the structural
prefilter under it went to 127.0.0.1:5432 while the rest of the query went to
the server the caller asked for.

Both database entry points are replaced with recorders, so these checks need no
Postgres, no embedding server and no LLM.
"""

import os
import sys
import traceback

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import suql.sql_free_text_support.execute_free_text_sql as suql_module
from suql.sql_free_text_support.execute_free_text_sql import suql_execute

HOST = "10.99.0.7"
PORT = "6543"

TABLE_W_IDS = {"restaurants": "_id"}
COLUMNS = [("_id", "text"), ("rating", "int4"), ("reviews", "text")]


class _ConnectionRecorder:
    """Stands in for both `execute_sql` and `execute_sql_with_column_info`,
    recording the (host, port) each call was routed to."""

    def __init__(self):
        self.calls = []

    def execute_sql(self, sql_query, database, host="127.0.0.1", port="5432", **kwargs):
        self.calls.append((host, port, str(sql_query)))
        return [], [], 0.0

    def execute_sql_with_column_info(
        self, sql_query, database, host="127.0.0.1", port="5432", **kwargs
    ):
        self.calls.append((host, port, str(sql_query)))
        return [], list(COLUMNS)


def _assert_all_connections_use_host_and_port(query):
    recorder = _ConnectionRecorder()
    saved = (suql_module.execute_sql, suql_module.execute_sql_with_column_info)
    suql_module.execute_sql = recorder.execute_sql
    suql_module.execute_sql_with_column_info = recorder.execute_sql_with_column_info
    try:
        suql_execute(
            query,
            TABLE_W_IDS,
            "restaurants",
            host=HOST,
            port=PORT,
            disable_try_catch=True,
            disable_retriever=True,
        )
    finally:
        suql_module.execute_sql, suql_module.execute_sql_with_column_info = saved

    assert recorder.calls, "expected the compiler to open at least one connection"
    elsewhere = [(h, p, q[:60]) for h, p, q in recorder.calls if (h, p) != (HOST, PORT)]
    assert not elsewhere, "these connections did not use {}:{} - {}".format(
        HOST, PORT, elsewhere
    )


# ---------- unit checks -------------------------------------------------------

def check_free_text_where_clause():
    _assert_all_connections_use_host_and_port(
        "SELECT _id FROM restaurants "
        "WHERE answer(reviews, 'is it quiet?') = 'Yes' LIMIT 1;"
    )


def check_free_text_and_structural_predicate():
    _assert_all_connections_use_host_and_port(
        "SELECT _id FROM restaurants "
        "WHERE rating = 5 AND answer(reviews, 'is it quiet?') = 'Yes' LIMIT 1;"
    )


def check_free_text_or_structural_predicate():
    _assert_all_connections_use_host_and_port(
        "SELECT _id FROM restaurants "
        "WHERE rating = 5 OR answer(reviews, 'is it quiet?') = 'Yes' LIMIT 1;"
    )


def check_structural_only_query():
    _assert_all_connections_use_host_and_port(
        "SELECT _id FROM restaurants WHERE rating = 5 LIMIT 1;"
    )


def check_free_text_in_projection():
    _assert_all_connections_use_host_and_port(
        "SELECT answer(reviews, 'is it quiet?') = 'Yes' AS quiet "
        "FROM restaurants WHERE rating = 5 LIMIT 1;"
    )


UNIT_CHECKS = [
    check_free_text_where_clause,
    check_free_text_and_structural_predicate,
    check_free_text_or_structural_predicate,
    check_structural_only_query,
    check_free_text_in_projection,
]


# ---------- runner ------------------------------------------------------------

def main():
    failures = []
    for check in UNIT_CHECKS:
        try:
            check()
            print("[PASS] {}".format(check.__name__))
        except Exception:
            print("[FAIL] {}".format(check.__name__))
            traceback.print_exc()
            failures.append(check.__name__)

    if failures:
        print("\n{} failed: {}".format(len(failures), failures))
        sys.exit(1)
    print("\nall checks passed")


if __name__ == "__main__":
    main()
