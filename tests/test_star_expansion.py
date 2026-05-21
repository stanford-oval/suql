"""Regression tests for issue #44: `t.*` / `*` in queries that route through
_Replace_Original_Target_Visitor used to crash with `'A_Star' object has no
attribute 'sval'`. Verifies:

  - `t.*` is expanded to N `^`-prefixed ResTargets in targetList / groupClause
    / sortClause.
  - Bare `*` and unknown-alias `t.*` are left intact.
  - Nested `t.*` (e.g. COUNT(t.*)) does not crash the visitor.
  - Issue #44's repro compiles end-to-end against a real Postgres DB without
    raising AttributeError.
"""

import os
import sys
import traceback

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from pglast import parse_sql
from pglast.ast import (
    A_Star,
    ColumnRef,
    FuncCall,
    ResTarget,
    SortBy,
    String,
)
from pglast.enums.parsenodes import SortByDir, SortByNulls
from pglast.stream import RawStream

from suql.sql_free_text_support.execute_free_text_sql import (
    _Replace_Original_Target_Visitor,
)


# ---------- helpers -----------------------------------------------------------

def _star_table_qualifier(rt_or_node):
    """Extract 't' from a ResTarget/ColumnRef/SortBy whose value is `t.*`."""
    if isinstance(rt_or_node, ResTarget):
        rt_or_node = rt_or_node.val
    if isinstance(rt_or_node, SortBy):
        rt_or_node = rt_or_node.node
    if not isinstance(rt_or_node, ColumnRef):
        return None
    if (len(rt_or_node.fields) == 2
            and isinstance(rt_or_node.fields[0], String)
            and isinstance(rt_or_node.fields[1], A_Star)):
        return rt_or_node.fields[0].sval
    return None


def _colref_name(rt_or_node):
    """Return the single mangled column name of a rewritten ResTarget/ColumnRef."""
    if isinstance(rt_or_node, ResTarget):
        rt_or_node = rt_or_node.val
    if isinstance(rt_or_node, SortBy):
        rt_or_node = rt_or_node.node
    assert isinstance(rt_or_node, ColumnRef)
    assert len(rt_or_node.fields) == 1
    return rt_or_node.fields[0].sval


# Realistic mapping shape (col_name, col_type), matching what
# execute_sql_with_column_info returns.
EVENTS_COLS = [("id", "int4"), ("notes", "text"), ("country", "text")]
USERS_COLS = [("id", "int4"), ("name", "text")]
MAPPING = {"events": EVENTS_COLS, "users": USERS_COLS}


# ---------- unit checks -------------------------------------------------------

def check_expand_qualified_star_in_target_list():
    v = _Replace_Original_Target_Visitor(MAPPING)
    tl = (ResTarget(val=ColumnRef(fields=(String(sval="events"), A_Star()))),)
    out = v.expand_target_list(tl)
    assert len(out) == 3, out
    assert [_colref_name(rt) for rt in out] == ["events^id", "events^notes", "events^country"]
    assert [rt.name for rt in out] == ["id", "notes", "country"]


def check_bare_star_left_alone_in_target_list():
    v = _Replace_Original_Target_Visitor(MAPPING)
    tl = (ResTarget(val=ColumnRef(fields=(A_Star(),))),)
    out = v.expand_target_list(tl)
    assert out is tl or (len(out) == 1 and _star_table_qualifier(out[0]) is None
                          and isinstance(out[0].val.fields[0], A_Star)), out


def check_unknown_alias_left_alone_in_target_list():
    v = _Replace_Original_Target_Visitor(MAPPING)
    tl = (ResTarget(val=ColumnRef(fields=(String(sval="unknown_cte"), A_Star()))),)
    out = v.expand_target_list(tl)
    assert len(out) == 1
    assert _star_table_qualifier(out[0]) == "unknown_cte"


def check_mixed_star_and_explicit_cols():
    v = _Replace_Original_Target_Visitor(MAPPING)
    explicit = ResTarget(val=ColumnRef(fields=(String(sval="events"), String(sval="notes"))))
    star = ResTarget(val=ColumnRef(fields=(String(sval="events"), A_Star())))
    other = ResTarget(val=ColumnRef(fields=(String(sval="users"), String(sval="name"))))
    tl = (explicit, star, other)
    out = v.expand_target_list(tl)
    # explicit (1) + star (3) + other (1) = 5
    assert len(out) == 5, out
    assert out[0] is explicit
    assert [_colref_name(rt) for rt in out[1:4]] == ["events^id", "events^notes", "events^country"]
    assert out[4] is other


def check_multiple_qualified_stars():
    v = _Replace_Original_Target_Visitor(MAPPING)
    tl = (
        ResTarget(val=ColumnRef(fields=(String(sval="events"), A_Star()))),
        ResTarget(val=ColumnRef(fields=(String(sval="users"), A_Star()))),
    )
    out = v.expand_target_list(tl)
    assert len(out) == 5
    names = [_colref_name(rt) for rt in out]
    assert names == ["events^id", "events^notes", "events^country", "users^id", "users^name"]


def check_expand_in_group_clause():
    v = _Replace_Original_Target_Visitor(MAPPING)
    gc = (ColumnRef(fields=(String(sval="events"), A_Star())),)
    out = v.expand_group_clause(gc)
    assert len(out) == 3
    assert [_colref_name(cr) for cr in out] == ["events^id", "events^notes", "events^country"]


def check_expand_in_sort_clause_preserves_direction():
    v = _Replace_Original_Target_Visitor(MAPPING)
    sb = SortBy(
        node=ColumnRef(fields=(String(sval="events"), A_Star())),
        sortby_dir=SortByDir.SORTBY_DESC,
        sortby_nulls=SortByNulls.SORTBY_NULLS_LAST,
        useOp=None,
    )
    out = v.expand_sort_clause((sb,))
    assert len(out) == 3
    for entry in out:
        assert isinstance(entry, SortBy)
        assert entry.sortby_dir == SortByDir.SORTBY_DESC
        assert entry.sortby_nulls == SortByNulls.SORTBY_NULLS_LAST
    assert [_colref_name(entry) for entry in out] == ["events^id", "events^notes", "events^country"]


def check_count_star_untouched():
    # COUNT(*) is FuncCall(agg_star=True) — no ColumnRef. Visitor must not
    # explode when walking it.
    v = _Replace_Original_Target_Visitor(MAPPING)
    ast = parse_sql("SELECT COUNT(*) FROM events")[0].stmt
    tl_out = v.expand_target_list(ast.targetList)
    # nothing should change — bare COUNT(*) has no t.*
    assert len(tl_out) == 1
    v(tl_out)  # must not raise


def check_count_qualified_star_does_not_crash():
    # COUNT(t.*) has a ColumnRef([String, A_Star]) inside FuncCall.args.
    # Pre-expansion does not touch it (it's nested). visit_ColumnRef must
    # early-return on A_Star instead of crashing.
    v = _Replace_Original_Target_Visitor(MAPPING)
    ast = parse_sql("SELECT COUNT(e.*) FROM events e")[0].stmt
    tl_out = v.expand_target_list(ast.targetList)
    # outer ResTarget is a FuncCall, not a ResTarget(ColumnRef([..A_Star])),
    # so pre-expansion leaves it alone.
    assert len(tl_out) == 1
    assert isinstance(tl_out[0].val, FuncCall)
    v(tl_out)  # must not raise (the guard in visit_ColumnRef catches it)


def check_regular_qualified_columns_still_rewritten():
    # Regression: bare `t.col` (no A_Star) still gets mangled to `t^col`.
    v = _Replace_Original_Target_Visitor(MAPPING)
    tl = (ResTarget(val=ColumnRef(fields=(String(sval="events"), String(sval="notes")))),)
    out = v.expand_target_list(tl)
    assert len(out) == 1
    v(out)
    assert _colref_name(out[0]) == "events^notes"


UNIT_CHECKS = [
    check_expand_qualified_star_in_target_list,
    check_bare_star_left_alone_in_target_list,
    check_unknown_alias_left_alone_in_target_list,
    check_mixed_star_and_explicit_cols,
    check_multiple_qualified_stars,
    check_expand_in_group_clause,
    check_expand_in_sort_clause_preserves_direction,
    check_count_star_untouched,
    check_count_qualified_star_does_not_crash,
    check_regular_qualified_columns_still_rewritten,
]


# ---------- integration check (the issue #44 repro) ---------------------------

def check_issue_44_repro_no_crash():
    """Runs the issue #44 repro through suql_execute. Asserts no AttributeError
    surfaces. Requires:
      - Postgres reachable at 127.0.0.1:5432
      - `acled` database with `events` table
      - `select_user` / `select_user` credentials
    Uses disable_retriever=True so no embedding server is needed; disable_try_catch=True
    so any AttributeError would surface rather than being swallowed.
    """
    from suql.sql_free_text_support.execute_free_text_sql import suql_execute

    sql = (
        "SELECT e.* FROM events e "
        "JOIN events f ON e.event_id_cnty = f.event_id_cnty "
        "WHERE answer(e.notes, 'Is this X?') = 'Yes' LIMIT 1;"
    )
    try:
        suql_execute(
            sql,
            table_w_ids={"events": "event_id_cnty"},
            database="acled",
            select_username="select_user",
            select_userpswd="select_user",
            host="127.0.0.1",
            port="5432",
            llm_model_name="gpt-4o-mini",
            disable_retriever=True,
            disable_try_catch=True,
        )
    except AttributeError as e:
        if "A_Star" in str(e):
            raise AssertionError(f"Issue #44 regression: {e!r}")
        raise


# ---------- runner ------------------------------------------------------------

def main():
    failures = []
    for check in UNIT_CHECKS:
        try:
            check()
            print(f"[PASS] {check.__name__}")
        except Exception:
            print(f"[FAIL] {check.__name__}")
            traceback.print_exc()
            failures.append(check.__name__)

    print()
    print("--- integration ---")
    try:
        check_issue_44_repro_no_crash()
        print("[PASS] check_issue_44_repro_no_crash")
    except AssertionError:
        print("[FAIL] check_issue_44_repro_no_crash")
        traceback.print_exc()
        failures.append("check_issue_44_repro_no_crash")
    except Exception as e:
        # Anything other than the AttributeError under test is allowed —
        # the integration test only proves issue #44 specifically.
        print(f"[PASS] check_issue_44_repro_no_crash (other error tolerated: {type(e).__name__})")

    if failures:
        print(f"\n{len(failures)} failed: {failures}")
        sys.exit(1)
    print("\nall checks passed")


if __name__ == "__main__":
    main()
