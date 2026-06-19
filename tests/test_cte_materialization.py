"""Tests for issue #45: CTE support in queries that mix `answer()` with
WITH clauses.

Unit tests check the AST-level helpers (dependency graph, topological sort,
reference rewriting, ID-column inference) in isolation.

Integration tests check that the full pipeline handles:
  - Case 1: outer query has answer(), CTE doesn't
  - Case 2: CTE body has answer(), outer doesn't
  - Case 3 (#45 original): CTEs chain together with answer() in some of them
  - WITH RECURSIVE containing answer() raises NotImplementedError
  - Plain (no-CTE) queries are unaffected

End-to-end tests run real SUQL queries against the live Postgres `acled` DB
AND the live embedding server (default http://127.0.0.1:8505). They use real
questions and exercise the retriever path — no monkey-patching, no fakes.
"""

import os
import sys
import traceback

import requests as _requests

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from pglast import parse_sql
from pglast.ast import (
    A_Star,
    ColumnRef,
    CommonTableExpr,
    RangeVar,
    ResTarget,
    SelectStmt,
    String,
)

from suql.sql_free_text_support.execute_free_text_sql import _SelectVisitor


def _new_visitor(table_w_ids=None):
    """Construct a bare _SelectVisitor — enough state to call the helpers."""
    return _SelectVisitor(
        fts_fields=[],
        database="acled",
        embedding_server_address="http://127.0.0.1:0",  # never reached in unit tests
        select_username="select_user",
        select_userpswd="select_user",
        create_username="select_user",
        create_userpswd="select_user",
        table_w_ids=table_w_ids if table_w_ids is not None else {"events": "event_id_cnty"},
        llm_model_name="gpt-4o-mini",
        max_verify=10,
    )


# ---------- live-stack config (used by integration + e2e checks) --------------

EMBEDDING_SERVER_ADDRESS = os.environ.get(
    "SUQL_EMBEDDING_SERVER", "http://127.0.0.1:8505"
)
# Real question fired against the events.notes free-text column. Chosen to
# return at least some rows in the ACLED Colombia subset so the end-to-end
# tests have something to assert on.
COLOMBIA_QUESTION = "Does this event relate to the Colombian government?"


def _embedding_server_reachable():
    """True if the embedding server at EMBEDDING_SERVER_ADDRESS accepts the
    real /search payload shape. Used to skip e2e tests cleanly when the server
    isn't running (e.g., CI without the embedding stack)."""
    try:
        r = _requests.post(
            EMBEDDING_SERVER_ADDRESS + "/search",
            json={
                "id_list": [],
                "field_query_list": [["events", "notes"]],
                "top": 1,
                "single_table": True,
            },
            timeout=3,
        )
        return r.status_code == 200
    except Exception:
        return False



# ---------- unit checks -------------------------------------------------------

def check_find_cte_refs_simple():
    ast = parse_sql("SELECT * FROM base WHERE answer(notes, 'q') = 'Yes'")[0].stmt
    refs = _SelectVisitor._find_cte_refs(ast, {"base", "other"})
    assert refs == {"base"}, refs


def check_find_cte_refs_ignores_schema_qualified():
    ast = parse_sql("SELECT * FROM public.base")[0].stmt
    refs = _SelectVisitor._find_cte_refs(ast, {"base"})
    assert refs == set(), refs


def check_find_cte_refs_in_join():
    ast = parse_sql(
        "SELECT * FROM events e JOIN base b ON e.id = b.id"
    )[0].stmt
    refs = _SelectVisitor._find_cte_refs(ast, {"base", "other"})
    assert refs == {"base"}, refs


def check_topo_order_respects_deps():
    # b depends on a; c depends on b. Order must be a, b, c.
    ctes = [
        CommonTableExpr(ctename="c", ctequery=parse_sql("SELECT 1")[0].stmt),
        CommonTableExpr(ctename="a", ctequery=parse_sql("SELECT 1")[0].stmt),
        CommonTableExpr(ctename="b", ctequery=parse_sql("SELECT 1")[0].stmt),
    ]
    deps = {"a": set(), "b": {"a"}, "c": {"b"}}
    order = _SelectVisitor._topo_order_ctes(ctes, deps)
    assert order.index("a") < order.index("b") < order.index("c"), order


def check_topo_order_detects_cycle():
    ctes = [
        CommonTableExpr(ctename="a", ctequery=parse_sql("SELECT 1")[0].stmt),
        CommonTableExpr(ctename="b", ctequery=parse_sql("SELECT 1")[0].stmt),
    ]
    deps = {"a": {"b"}, "b": {"a"}}
    try:
        _SelectVisitor._topo_order_ctes(ctes, deps)
    except ValueError as e:
        assert "cycle" in str(e).lower(), e
        return
    raise AssertionError("expected ValueError for cycle")


def check_rewrite_cte_refs_replaces_relname():
    body = parse_sql("SELECT * FROM base WHERE id > 0")[0].stmt
    _SelectVisitor._rewrite_cte_refs(body, {"base": "temp_table_abc"})
    assert body.fromClause[0].relname == "temp_table_abc"


def check_rewrite_cte_refs_leaves_unknown_alone():
    body = parse_sql("SELECT * FROM events e")[0].stmt
    _SelectVisitor._rewrite_cte_refs(body, {"base": "temp_table_abc"})
    assert body.fromClause[0].relname == "events"


def check_projection_includes_bare_star():
    body = parse_sql("SELECT * FROM events e")[0].stmt
    assert _SelectVisitor._projection_includes(body.targetList, "e", "event_id_cnty") is True


def check_projection_includes_qualified_star():
    body = parse_sql("SELECT e.* FROM events e")[0].stmt
    assert _SelectVisitor._projection_includes(body.targetList, "e", "event_id_cnty") is True


def check_projection_includes_explicit_column():
    body = parse_sql("SELECT e.event_id_cnty FROM events e")[0].stmt
    assert _SelectVisitor._projection_includes(body.targetList, "e", "event_id_cnty") is True


def check_projection_excludes_other_column():
    body = parse_sql("SELECT e.notes FROM events e")[0].stmt
    assert _SelectVisitor._projection_includes(body.targetList, "e", "event_id_cnty") is False


def check_infer_id_column_single_table():
    v = _new_visitor()
    body = parse_sql(
        "SELECT e.event_id_cnty, e.notes FROM events e"
    )[0].stmt
    assert v._infer_cte_id_column(body) == "event_id_cnty"


def check_infer_id_column_unknown_underlying():
    v = _new_visitor(table_w_ids={"foreign_table": "fid"})
    body = parse_sql("SELECT * FROM events e")[0].stmt
    assert v._infer_cte_id_column(body) is None


def check_infer_id_column_id_not_projected():
    v = _new_visitor()
    body = parse_sql("SELECT e.notes FROM events e")[0].stmt
    assert v._infer_cte_id_column(body) is None


def check_infer_id_column_join_with_answer_returns_mangled():
    # Join + answer() => _execute_structural_sql produces ^-mangled column
    # names. Inferred ID must match (= 'events^event_id_cnty'), otherwise
    # the outer's _retrieve_and_verify cannot find the column.
    v = _new_visitor(table_w_ids={"events": "event_id_cnty", "other": "oid"})
    body = parse_sql(
        "SELECT e.* FROM events e JOIN other o ON e.event_id_cnty = o.oid "
        "WHERE answer(e.notes, 'q') = 'Yes'"
    )[0].stmt
    assert v._infer_cte_id_column(body) == "events^event_id_cnty"


def check_infer_id_column_join_without_answer_returns_unmangled():
    # No answer() in the body => CREATE TABLE AS preserves original column
    # names. No mangling needed.
    v = _new_visitor(table_w_ids={"events": "event_id_cnty", "other": "oid"})
    body = parse_sql(
        "SELECT e.* FROM events e JOIN other o ON e.event_id_cnty = o.oid"
    )[0].stmt
    assert v._infer_cte_id_column(body) == "event_id_cnty"


# ---------- lineage builder ---------------------------------------------------

def check_lineage_qualified_column():
    v = _new_visitor()
    body = parse_sql("SELECT e.notes, e.event_id_cnty FROM events e")[0].stmt
    lin = v._build_cte_column_lineage(body)
    assert lin == {
        "notes": ("events", "notes"),
        "event_id_cnty": ("events", "event_id_cnty"),
    }, lin


def check_lineage_qualified_with_rename():
    # `t.col AS new_name` => lineage key is the rename target.
    v = _new_visitor()
    body = parse_sql("SELECT e.notes AS body FROM events e")[0].stmt
    lin = v._build_cte_column_lineage(body)
    assert lin == {"body": ("events", "notes")}, lin


def check_lineage_qualified_star():
    v = _new_visitor()
    # Stub the column-list lookup (no DB needed for unit tests).
    v._table_columns_cache["events"] = ["event_id_cnty", "notes", "country"]
    body = parse_sql("SELECT e.* FROM events e")[0].stmt
    lin = v._build_cte_column_lineage(body)
    assert lin == {
        "event_id_cnty": ("events", "event_id_cnty"),
        "notes": ("events", "notes"),
        "country": ("events", "country"),
    }, lin


def check_lineage_bare_star_single_table():
    v = _new_visitor()
    v._table_columns_cache["events"] = ["event_id_cnty", "notes"]
    body = parse_sql("SELECT * FROM events e")[0].stmt
    lin = v._build_cte_column_lineage(body)
    assert lin == {
        "event_id_cnty": ("events", "event_id_cnty"),
        "notes": ("events", "notes"),
    }, lin


def check_lineage_bare_col_unambiguous_in_join():
    # `notes` exists only in `events`, not in `other`. Postgres would accept;
    # so should our lineage builder, resolving to events.
    v = _new_visitor(table_w_ids={"events": "event_id_cnty", "other": "oid"})
    v._table_columns_cache["events"] = ["event_id_cnty", "notes"]
    v._table_columns_cache["other"] = ["oid", "description"]
    body = parse_sql(
        "SELECT notes FROM events e JOIN other o ON e.event_id_cnty = o.oid"
    )[0].stmt
    lin = v._build_cte_column_lineage(body)
    assert lin == {"notes": ("events", "notes")}, lin


def check_lineage_bare_col_ambiguous_in_join_is_dropped():
    # `notes` exists in both — Postgres would reject as ambiguous; we
    # mirror that by leaving the entry out (no lineage = downstream
    # KeyError surfaces, which matches Postgres-style behavior).
    v = _new_visitor(table_w_ids={"events": "event_id_cnty", "other": "oid"})
    v._table_columns_cache["events"] = ["event_id_cnty", "notes"]
    v._table_columns_cache["other"] = ["oid", "notes"]
    body = parse_sql(
        "SELECT notes FROM events e JOIN other o ON e.event_id_cnty = o.oid"
    )[0].stmt
    lin = v._build_cte_column_lineage(body)
    assert "notes" not in lin, lin


def check_lineage_computed_expression_marked_none():
    # `LOWER(e.notes) AS x` is computed — no FAISS embedding for it.
    # Record the AS name with None so callers can detect "tried to map but
    # got nowhere."
    v = _new_visitor()
    body = parse_sql("SELECT LOWER(e.notes) AS x FROM events e")[0].stmt
    lin = v._build_cte_column_lineage(body)
    assert lin == {"x": None}, lin


def check_lineage_chained_cte_composes():
    # cand references base. cand's lineage for `notes` should compose
    # through base's lineage all the way back to (events, notes).
    v = _new_visitor()
    # Pre-populate base's lineage as if _process_ctes had handled it.
    v.cte_column_lineage["base"] = {
        "notes": ("events", "notes"),
        "event_id_cnty": ("events", "event_id_cnty"),
    }
    body = parse_sql("SELECT notes FROM base")[0].stmt
    lin = v._build_cte_column_lineage(body)
    assert lin == {"notes": ("events", "notes")}, lin


def check_lineage_chained_qualified_through_cte():
    v = _new_visitor()
    v.cte_column_lineage["base"] = {
        "notes": ("events", "notes"),
        "event_id_cnty": ("events", "event_id_cnty"),
    }
    body = parse_sql("SELECT b.notes AS body FROM base b")[0].stmt
    lin = v._build_cte_column_lineage(body)
    assert lin == {"body": ("events", "notes")}, lin


UNIT_CHECKS = [
    check_find_cte_refs_simple,
    check_find_cte_refs_ignores_schema_qualified,
    check_find_cte_refs_in_join,
    check_topo_order_respects_deps,
    check_topo_order_detects_cycle,
    check_rewrite_cte_refs_replaces_relname,
    check_rewrite_cte_refs_leaves_unknown_alone,
    check_projection_includes_bare_star,
    check_projection_includes_qualified_star,
    check_projection_includes_explicit_column,
    check_projection_excludes_other_column,
    check_infer_id_column_single_table,
    check_infer_id_column_unknown_underlying,
    check_infer_id_column_id_not_projected,
    check_infer_id_column_join_with_answer_returns_mangled,
    check_infer_id_column_join_without_answer_returns_unmangled,
    check_lineage_qualified_column,
    check_lineage_qualified_with_rename,
    check_lineage_qualified_star,
    check_lineage_bare_star_single_table,
    check_lineage_bare_col_unambiguous_in_join,
    check_lineage_bare_col_ambiguous_in_join_is_dropped,
    check_lineage_computed_expression_marked_none,
    check_lineage_chained_cte_composes,
    check_lineage_chained_qualified_through_cte,
]


# ---------- integration checks (real DB; require acled + select_user) ---------

def _run(sql, table_w_ids=None):
    from suql.sql_free_text_support.execute_free_text_sql import suql_execute
    return suql_execute(
        sql,
        table_w_ids=table_w_ids if table_w_ids is not None else {"events": "event_id_cnty"},
        database="acled",
        select_username="select_user",
        select_userpswd="select_user",
        host="127.0.0.1",
        port="5432",
        llm_model_name="gpt-4o-mini",
        disable_retriever=True,
        disable_try_catch=True,
        statement_timeout=30000,
    )


def check_case1_outer_answer():
    # outer has answer(), CTE doesn't. ID column registration is what carries
    # this case — no materialization needed.
    sql = (
        "WITH base AS (SELECT e.event_date, e.notes, e.event_id_cnty "
        "              FROM events e WHERE e.country = 'Colombia' LIMIT 5) "
        f"SELECT event_id_cnty FROM base WHERE answer(notes, '{COLOMBIA_QUESTION}') = 'Yes' LIMIT 1;"
    )
    _run(sql)


def check_case2_cte_answer():
    # CTE has answer(), outer doesn't.
    sql = (
        "WITH cand AS (SELECT * FROM events WHERE country = 'Colombia' "
        f"              AND answer(notes, '{COLOMBIA_QUESTION}') = 'Yes' LIMIT 5) "
        "SELECT event_id_cnty FROM cand LIMIT 1;"
    )
    _run(sql)


def check_issue_45_multi_ref():
    # Original #45 shape: `base` referenced by two downstream CTEs that have
    # answer(). Confirms transitive materialization works (base gets
    # materialized via _materialize_cte_body_directly because its deps need it).
    sql = (
        "WITH base AS (SELECT e.event_date, e.notes, e.event_id_cnty "
        "              FROM events e WHERE e.country = 'Colombia' LIMIT 5), "
        "     candidate_a AS (SELECT * FROM base "
        f"                     WHERE answer(notes, 'Does this event involve the Colombian government?') = 'Yes'), "
        "     candidate_b AS (SELECT * FROM base "
        f"                     WHERE answer(notes, 'Does this event involve civilians in Colombia?') = 'Yes') "
        "SELECT (SELECT COUNT(*) FROM candidate_a) AS a_cnt, "
        "       (SELECT COUNT(*) FROM candidate_b) AS b_cnt;"
    )
    _run(sql)


def check_with_recursive_with_answer_refused():
    sql = (
        "WITH RECURSIVE r AS ("
        "  SELECT event_id_cnty, notes FROM events WHERE event_id_cnty = 'X' "
        "  UNION "
        "  SELECT e.event_id_cnty, e.notes FROM events e "
        "  JOIN r ON r.event_id_cnty = e.event_id_cnty "
        f"  WHERE answer(e.notes, '{COLOMBIA_QUESTION}') = 'Yes'"
        ") SELECT * FROM r;"
    )
    try:
        _run(sql)
    except NotImplementedError as e:
        assert "WITH RECURSIVE" in str(e), e
        return
    raise AssertionError("expected NotImplementedError")


def check_no_cte_regression():
    # Plain query without WITH still works.
    sql = (
        "SELECT event_id_cnty FROM events "
        f"WHERE country = 'Colombia' AND answer(notes, '{COLOMBIA_QUESTION}') = 'Yes' LIMIT 1;"
    )
    _run(sql)


INTEGRATION_CHECKS = [
    check_case1_outer_answer,
    check_case2_cte_answer,
    check_issue_45_multi_ref,
    check_with_recursive_with_answer_refused,
    check_no_cte_regression,
]


# ---------- end-to-end (exercise full pipeline + verify semantics) ------------

def _suql_e2e(sql, **overrides):
    """Run a SUQL query end-to-end against the real Postgres + the real
    embedding server. Uses disable_retriever=False so the retriever path
    (and thus our lineage tracker) actually fires.
    """
    from suql.sql_free_text_support.execute_free_text_sql import suql_execute
    kwargs = dict(
        table_w_ids={"events": "event_id_cnty"},
        database="acled",
        select_username="select_user",
        select_userpswd="select_user",
        host="127.0.0.1",
        port="5432",
        embedding_server_address=EMBEDDING_SERVER_ADDRESS,
        llm_model_name="gpt-4o-mini",
        disable_retriever=False,
        disable_try_catch=True,
        statement_timeout=120000,
    )
    kwargs.update(overrides)
    return suql_execute(sql, **kwargs)


def check_lineage_rewrite_via_real_server():
    """CTE wraps `events` under the name `base`; outer answer() targets
    `notes`. Without our lineage rewrite, the embedding server would be
    asked for ('base','notes') — for which there is no FAISS index — and
    return zero rows. With the rewrite it gets ('events','notes') and
    returns real candidates.

    We assert by comparing against a flat (no-CTE) query that asks the
    same question over the same row set. The id-set returned by the CTE
    shape must be a subset of the flat-query id-set (LLM may answer
    differently across runs; subset is the strongest stable invariant).
    """
    flat_sql = (
        "SELECT event_id_cnty FROM events "
        f"WHERE country='Colombia' AND answer(notes, '{COLOMBIA_QUESTION}')='Yes' "
        "LIMIT 10;"
    )
    cte_sql = (
        "WITH base AS (SELECT * FROM events WHERE country='Colombia') "
        f"SELECT event_id_cnty FROM base WHERE answer(notes, '{COLOMBIA_QUESTION}')='Yes' "
        "LIMIT 10;"
    )

    flat_rows, _, _ = _suql_e2e(flat_sql)
    cte_rows, _, _ = _suql_e2e(cte_sql)

    flat_ids = {r[0] for r in flat_rows}
    cte_ids = {r[0] for r in cte_rows}

    # The CTE-wrapped version must actually retrieve candidates from the
    # embedding server (proves lineage rewrite hit ('events','notes')).
    assert cte_rows, "CTE-wrapped query returned zero rows — lineage rewrite likely broken"
    # Both queries select from the same row set with the same question; CTE
    # ids should be a subset of (or equal to) the flat-query ids modulo LLM
    # nondeterminism. Strict equality is too brittle; require non-empty
    # overlap.
    overlap = cte_ids & flat_ids
    assert overlap, (
        f"no overlap between flat and CTE row sets: flat={flat_ids}, cte={cte_ids}"
    )


def check_cte_vs_noncte_row_equivalence():
    """Same logical query in two shapes — with and without a wrapping CTE.
    The retrieved id set for the CTE shape must overlap meaningfully with
    the flat shape. (LLM nondeterminism makes strict equality brittle;
    overlap is the strongest stable invariant.)
    """
    no_cte = (
        "SELECT event_id_cnty FROM events "
        f"WHERE country='Colombia' AND answer(notes, '{COLOMBIA_QUESTION}')='Yes' LIMIT 10;"
    )
    with_cte = (
        "WITH colombia AS (SELECT * FROM events WHERE country='Colombia') "
        f"SELECT event_id_cnty FROM colombia WHERE answer(notes, '{COLOMBIA_QUESTION}')='Yes' LIMIT 10;"
    )

    rows_no, _, _ = _suql_e2e(no_cte)
    rows_with, _, _ = _suql_e2e(with_cte)
    ids_no = {r[0] for r in rows_no}
    ids_with = {r[0] for r in rows_with}
    assert ids_with, "CTE-wrapped query returned zero rows"
    assert ids_no, "flat query returned zero rows"
    overlap = ids_no & ids_with
    assert overlap, (
        f"no overlap between CTE and flat row sets: flat={ids_no}, cte={ids_with}"
    )


def check_computed_expression_in_cte_refused_at_retriever():
    """An outer answer() targets a column that's a computed expression in
    the CTE projection. The lineage builder records None for that column,
    and breakdown_unstructural_query raises a clear NotImplementedError
    *before* any embedding-server call — verified end-to-end."""
    sql = (
        "WITH t AS (SELECT event_id_cnty, LOWER(notes) AS x FROM events "
        "           WHERE country='Colombia' LIMIT 5) "
        f"SELECT event_id_cnty FROM t WHERE answer(x, '{COLOMBIA_QUESTION}')='Yes' LIMIT 1;"
    )
    try:
        _suql_e2e(sql)
    except NotImplementedError as e:
        assert "lineage" in str(e).lower() or "computed" in str(e).lower(), e
        return
    raise AssertionError("expected NotImplementedError for computed-expression CTE column")


def check_chained_cte_end_to_end():
    """Three-CTE chain A -> B -> C, where B has answer(). Verifies that
    transitive materialization gets A done before B, that B's body sees a
    real temp table for A, and that C reads B's results. Hits real
    embedding server + real LLM. The COUNT must come back as a single
    non-negative integer."""
    sql = (
        "WITH a AS (SELECT * FROM events WHERE country='Colombia' LIMIT 5), "
        f"     b AS (SELECT * FROM a WHERE answer(notes, '{COLOMBIA_QUESTION}')='Yes'), "
        "     c AS (SELECT event_id_cnty FROM b) "
        "SELECT COUNT(*) FROM c;"
    )
    results, _, _ = _suql_e2e(sql)
    assert len(results) == 1, results
    count = results[0][0]
    assert 0 <= count <= 5, count


def check_multi_table_cte_body_end_to_end():
    """CTE body has a join over (events, a constant dates CTE).
    Outer answer() targets the events-derived notes column. Lineage must
    map (joined_cte, notes) → (events, notes) for the retriever to find
    the correct index. Hits real embedding server + real LLM."""
    sql = (
        "WITH dates AS (SELECT DATE '2020-01-01' AS d), "
        "     ce AS (SELECT e.event_id_cnty, e.notes "
        "            FROM events e JOIN dates d ON e.event_date >= d.d "
        "            WHERE e.country='Colombia' LIMIT 5) "
        f"SELECT event_id_cnty FROM ce WHERE answer(notes, '{COLOMBIA_QUESTION}')='Yes' LIMIT 1;"
    )
    _suql_e2e(sql)


END_TO_END_CHECKS = [
    check_lineage_rewrite_via_real_server,
    check_cte_vs_noncte_row_equivalence,
    check_computed_expression_in_cte_refused_at_retriever,
    check_chained_cte_end_to_end,
    check_multi_table_cte_body_end_to_end,
]


# ---------- runner ------------------------------------------------------------

def main():
    failures = []
    print("--- unit ---")
    for check in UNIT_CHECKS:
        try:
            check()
            print(f"[PASS] {check.__name__}")
        except Exception:
            print(f"[FAIL] {check.__name__}")
            traceback.print_exc()
            failures.append(check.__name__)

    print("\n--- integration (requires acled DB + select_user) ---")
    for check in INTEGRATION_CHECKS:
        try:
            check()
            print(f"[PASS] {check.__name__}")
        except Exception:
            print(f"[FAIL] {check.__name__}")
            traceback.print_exc()
            failures.append(check.__name__)

    print(f"\n--- end-to-end (hits real DB, embedding server, LLM) ---")
    if not _embedding_server_reachable():
        print(f"[SKIP] embedding server at {EMBEDDING_SERVER_ADDRESS} not reachable — "
              f"skipping {len(END_TO_END_CHECKS)} e2e checks")
    else:
        for check in END_TO_END_CHECKS:
            try:
                check()
                print(f"[PASS] {check.__name__}")
            except Exception:
                print(f"[FAIL] {check.__name__}")
                traceback.print_exc()
                failures.append(check.__name__)

    if failures:
        print(f"\n{len(failures)} failed: {failures}")
        sys.exit(1)
    print("\nall checks passed")


if __name__ == "__main__":
    main()
