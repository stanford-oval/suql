"""Regression for duplicate LLM requests during projection verification."""

import os
import sys
import threading
from collections import Counter

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from pglast import parse_sql

import suql.sql_free_text_support.execute_free_text_sql as suql_module


def check_projection_verifies_duplicate_documents_once():
    # The marked CTE from issue #47, with two rows sharing the same notes.
    node = parse_sql("""
        SELECT month,
          CASE WHEN answer(notes, 'Is this X?') = 'Yes'
                    AND (answer(actors_text, 'Is this Y?') = 'Yes'
                         OR answer(notes, 'Is this Y?') = 'Yes')
               THEN 1 ELSE 0 END AS flag_a,
          CASE WHEN (inter1 = 'State forces' OR inter2 = 'State forces')
                    AND answer(notes, 'Is this Z?') = 'Yes'
               THEN 1 ELSE 0 END AS flag_b
        FROM base
    """)[0].stmt
    predicates = suql_module._extract_projection_predicates(node)
    rows = [
        ("2024-01-01", "State forces", "", "event X Z", "event X Z"),
        ("2024-02-01", "", "State forces", "event X Z", "other actor"),
    ]
    columns = [
        (name, "text")
        for name in ("month", "inter1", "inter2", "notes", "actors_text")
    ]
    seen = []
    barrier = None
    original_map = suql_module._parallel_map
    original_llm = suql_module.llm_generate
    original_cache = suql_module._verified_res
    budget = suql_module._VerificationBudget(max_cost=None)
    budget_token = suql_module._verification_budget.set(budget)

    def synchronized_map(fcn, items, max_workers=None):
        nonlocal barrier
        barrier = threading.Barrier(len(items))
        return original_map(fcn, items, max_workers=len(items))

    def fake_llm(**kwargs):
        params = kwargs["prompt_parameter_values"]
        seen.append((params["document"], params["field"], params["query"], params["answer"]))
        # Keep all requests in flight until they have passed the memo lookup.
        barrier.wait(timeout=5)
        accepted = (params["document"], params["field"], params["query"]) in {
            ("event X Z", "notes", "Is this X?"),
            ("event X Z", "notes", "Is this Z?"),
            ("event X Z", "actors_text", "Is this Y?"),
        }
        return ("the answer is correct" if accepted else "incorrect",)

    try:
        suql_module._parallel_map = synchronized_map
        suql_module.llm_generate = fake_llm
        suql_module._verified_res = {}
        got, got_columns = suql_module._verify_projection_predicates(
            node, predicates, rows, columns, suql_module._VERIFICATION_MODEL_NAME
        )
    finally:
        suql_module._parallel_map = original_map
        suql_module.llm_generate = original_llm
        suql_module._verified_res = original_cache
        suql_module._verification_budget.reset(budget_token)

    assert len(got) == len(rows), got
    assert got_columns == columns + [
        (alias, "boolean") for alias in sorted(predicates)
    ], got_columns
    for row, output in zip(rows, got):
        assert output[:len(row)] == row, output
        expected = []
        for alias, _ in got_columns[len(columns):]:
            text_arg, query, _, _ = predicates[alias]
            field = text_arg.fields[-1].sval
            document = row[4 if field == "actors_text" else 3]
            expected.append((document, field, query) in {
                ("event X Z", "notes", "Is this X?"),
                ("event X Z", "notes", "Is this Z?"),
                ("event X Z", "actors_text", "Is this Y?"),
            })
        assert output[len(row):] == tuple(expected), output
    assert len(seen) == 5, "expected 5 distinct verifications, got {}".format(len(seen))
    assert Counter(seen) == Counter({
        ("event X Z", "notes", "Is this X?", "Yes"): 1,
        ("event X Z", "notes", "Is this Y?", "Yes"): 1,
        ("event X Z", "notes", "Is this Z?", "Yes"): 1,
        ("event X Z", "actors_text", "Is this Y?", "Yes"): 1,
        ("other actor", "actors_text", "Is this Y?", "Yes"): 1,
    }), seen
    assert budget.snapshot()["planned"] == 5, budget.snapshot()
    assert budget.snapshot()["verifications"] == 5, budget.snapshot()


if __name__ == "__main__":
    check_projection_verifies_duplicate_documents_once()
    print("[PASS] check_projection_verifies_duplicate_documents_once")
