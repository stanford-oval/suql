# Changelog

All notable changes to SUQL are documented in this file. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Fixed
- **`host` and `port` now reach the structural prefilter of a query with
  `answer()` in its WHERE clause.** `_analyze_SelectStmt` took no `host` or
  `port`, so that prefilter always went to 127.0.0.1:5432 while the temp table
  and the rewritten query went to the configured server. With the database
  anywhere else, such a query returned an empty result instead of raising,
  unless `disable_try_catch=True` was set.

## [1.1.10a4] - 2026-08-11

### Changed (behavioural — read before upgrading)
- **`answer(...)` verification is now capped at $1.00 per query by default.**
  Previously nothing bounded it. A query that used to run to completion while
  spending more than that now raises `SUQLCostLimitExceeded` instead of
  returning results. Pass `max_verification_cost=<usd>` to raise the ceiling,
  or `0` to restore the old unbounded behaviour. See the entry below for why
  the default is on rather than opt-in.

### Added
- **Per-query verification cost ceiling (`max_verification_cost`, default
  $1.00).** Nothing previously bounded what one query could spend verifying
  `answer(...)` predicates. A reported run ground for 25 hours on 4,326,211
  verification calls costing $101.51 without a single query completing — broad
  free-text predicates over a large candidate set. Parallelising verification
  made that faster, not cheaper.

  Enforced at `_verify`, the single point every compiler-side verification
  passes through, so it covers both the retriever-backed WHERE path and
  projection predicates. Two mechanisms:
  - *Stop*: accumulated spend is checked before every verification, so a query
    can never run past its ceiling.
  - *Refuse early*: after a small sample of real calls (25), the mean cost per
    verification is extrapolated over the number still planned; if the
    projection exceeds the ceiling the query is refused immediately. A plan of
    176,660 verifications is now refused in ~12 seconds having spent $0.0006,
    with a message giving the plan size, the measured rate, the projection and
    the limit.

  Exceeding the ceiling raises `SUQLCostLimitExceeded` (exported from `suql`)
  rather than returning a half-applied filter as if it were the complete
  answer. Cost is projected per verification *attempt*, so documents served
  from the `_verified_res` memo cache do not inflate the estimate; the planned
  count for a multi-predicate WHERE clause is an upper bound, so the projection
  errs toward refusing. Override per call, or process-wide with
  `SUQL_MAX_VERIFICATION_COST`; pass `0` to remove the ceiling.
  `max_verification_calls` (and `SUQL_MAX_VERIFICATION_CALLS`) caps the call
  count instead. `cache["_stats"]` now also reports `verifications` and
  `max_verification_cost`.

  Scope: this bounds spend made in the SUQL process. `answer()` left in a
  projection for Postgres to evaluate as the raw plpython3u UDF calls the
  free-text server directly and is not covered; that path is still only visible
  after the fact via `/stats/<query_id>`. The ceiling is also per
  `suql_execute` call — a pipeline issuing many queries needs its own aggregate
  budget on top.

### Fixed
- **CTE materialization dropped information the rest of the query needed.**
  Four distinct failures, all from materializing a CTE containing `answer()`
  and then losing something across the boundary. Reported together from one
  pipeline run in which 12/12 SUQL executions failed.
  - *The name a CTE is referred to by.* `_rewrite_cte_refs` swapped
    `RangeVar.relname` for the temp table without keeping the old name, so
    `SELECT base.event_id_cnty ... FROM base` became `... FROM temp_table_xxx`
    and Postgres rejected it with `missing FROM-clause entry for table "base"`.
    The same happened to an explicit alias (`FROM base b` … `SELECT b.*`),
    which `visit_SelectStmt` then overwrote with the temp table's own name. The
    referenced name is now preserved as an alias, and an existing alias is
    carried across instead of being replaced.
  - *The CTE's own output columns.* A CTE was treated as "materialized" if its
    FROM clause named a temp table — which is also true when it merely *reads*
    from an upstream CTE's temp table. Downstream references were then
    redirected to that upstream table, dropping every column the body computed
    (`column "russian_yes" does not exist`). Redirection now requires that the
    body actually reduce to `SELECT * FROM <temp table>`; when it does not, the
    body's own output is materialized first.
  - *The row ID.* A CTE that does not project its source table's ID column
    (`SELECT country, admin1, notes FROM events`) could not be registered in
    `table_w_ids`, and a downstream `answer()` failed with a bare
    `KeyError: 'temp_table_xxx'`. The ID is now added back to such a CTE before
    it is materialized. Only CTEs materialized as *input* to a later `answer()`
    are touched, so this does not change what `SELECT * FROM <cte>` returns.
  - *Cases that genuinely have no row ID* (a CTE that aggregates) now raise an
    actionable `NotImplementedError` naming the relation and the ID columns it
    could project, instead of a `KeyError` on an internal table name.
- **`answer()` comparisons outside the WHERE clause.**
  `CASE WHEN answer(notes, q) = 'Yes' THEN 1 ELSE 0 END` in a SELECT list
  returned 0 for rows the model had answered correctly, with no error — silently
  under-counting aggregates. Only `node.whereClause` was ever compiled, so a
  comparison in the projection was handed to Postgres, which ran the raw
  plpython3u UDF (whose prompt is open-ended: "Answer a question based on the
  following text") and byte-compared its free-form prose against the literal.
  `Yes. A nuclear power station is part of energy infrastructure.` is not
  `'Yes'`. It also varied run to run, because `prompt_continuation` forces
  `temperature=1` for the gpt-5 family regardless of the caller's request.
  The compiler now recognises `answer(...) <op> <literal>` in the projection,
  `HAVING`, `GROUP BY` and `ORDER BY`, verifies each one per row through the
  same path a WHERE predicate takes, and materializes the result as a synthetic
  boolean column. A bare `SELECT *` alongside such a predicate is expanded to
  the explicit column list so the synthetic column doesn't leak. Uses that want
  the model's prose (`SELECT answer(notes, 'which city?')`), comparisons against
  a non-literal, and calls wrapped in another function (`lower(answer(...))`)
  are left for Postgres as before; an aggregate in the text argument raises
  `NotImplementedError` rather than silently collapsing to one row.

- **`answer()` over a computed text expression (issue #50).** The text argument
  of a free-text function no longer has to be a bare column: `answer(COALESCE(a,
  '') || ' ' || COALESCE(b, ''), '...')`, `answer(lower(notes), '...')` and any
  other text-typed SQL expression now work, instead of tripping
  `assert len(field_lst) == 1` in `breakdown_unstructural_query`. The compiler
  projects the expression in the structural query under an internal
  `_suql_expr_<hash>` alias (stripped again before results are returned), so
  Postgres evaluates it once per row and the LLM verifies the expression's own
  value. Retrieval falls back to the FAISS indexes of the columns the
  expression reads from — an unindexed column is skipped with a warning, and
  when nothing is retrievable the compiler verifies every surviving row, or
  raises if that would exceed 1000 rows (pass `disable_retriever=True` to force
  it). The text argument and the question are now read positionally, so a
  string literal as the text argument no longer confuses the parser either.
- **Table-qualified projections with `answer()`.** `SELECT e.event_id_cnty FROM
  events e WHERE answer(...)` failed with `missing FROM-clause entry for table
  "events"`: the FROM clause is swapped for the temp table holding the verified
  rows, which left qualified references dangling. The temp table now carries
  the original table name as an alias.
- `answer()` on a NULL text value is now `False` instead of raising an
  `AssertionError` inside verification.

### Changed
- **Verification LLM calls are explicitly pooled.** Both the WHERE-side filter
  and the new projection-side path run their verifications through a thread pool
  sized by `max_verification_workers` (new `suql_execute(...)` parameter,
  default 32, overridable process-wide with `SUQL_MAX_VERIFICATION_WORKERS`).
  Previously `_parallel_filtering` constructed an unconfigured
  `ThreadPoolExecutor`, whose size is `min(32, cpu_count + 4)` — a function of
  the local core count, when the real ceiling is the LLM provider's rate limit.
  A projection predicate cannot prune, so it is verified on every output row;
  the calls are all mutually independent and go into one flat pool rather than
  being nested per row. Where the query's `LIMIT` cannot be changed by anything
  downstream (no `GROUP BY`/`HAVING`/`ORDER BY`/`DISTINCT`), it is pushed down
  ahead of verification, which bounds the number of LLM calls.

## [1.1.10a3] - 2026-05-13

### Added
- **`disable_retriever` parameter on `suql_execute(...)`.** Lightweight/demo
  mode: when `True`, the SUQL compiler skips the embedding-based retriever
  entirely and verifies every row that survives the structural prefilter with
  the LLM. No embedding server / FAISS index is required. Threaded through the
  compiler stack (`_SelectVisitor` → `_analyze_SelectStmt` → `_execute_and` →
  `_execute_free_text_queries` → `_retrieve_and_verify`). At the top of
  `suql_execute` a `NOTE:` is printed to stdout when the flag is on, explaining
  the cost characteristics (O(rows) LLM calls per `answer(...)` predicate).
  Useful for small tables and demos where running an embedding server is
  undesirable; pair with a `LIMIT` clause to bound cost.

### Changed
- Trimmed the `if __name__ == "__main__":` block in
  `execute_free_text_sql.py` (test-harness leftovers).

## [1.1.10a2] - 2026-04-25

### Changed
- **Default model is now `gpt-5.2` everywhere.** gpt-5.2 supports
  `reasoning_effort="none"`, which the gpt-5/gpt-5.4 family does not — in those
  models, even `reasoning_effort="minimal"` can allocate hidden reasoning tokens
  out of the response budget and cause `max_tokens=30-100` calls to return empty
  content. With every short-task path silently rejecting rows, queries like
  `WHERE answer(...) = 'yes'` returned `[]`.
- **Bumped LiteLLM lower bound** from `>=1.34.34` to `>=1.77.7` to guarantee
  full `reasoning_effort` support across the gpt-5 family.

### Added
- **`debug_log` parameter on `suql_execute(...)`.** Pass `debug_log=True` (or a
  path) to capture per-call input/output for every `llm_generate`, `/answer`,
  and `/summary` invocation made on behalf of the query. Implemented via a
  per-`query_id` registry on the free-text server (`POST /debug`), so no
  plpython3u UDF changes are required. Useful for diagnosing silent rejections,
  prompt issues, or model-output format drift.

## [1.1.10a1] - 2026-04-17

### Added
- Cost tracking for `suql_execute(...)`: each call returns aggregated
  `cost`/`calls` stats under `cache["_stats"]`.
- `statement_timeout` parameter exposed on `suql_execute(...)`.

## [1.1.9] - 2026-04-16

### Added
- SUQL Python client and a basic SUQL REPL loop.

### Fixed
- gpt-5 compatibility: force `temperature=1` and drop unsupported sampling
  params for the gpt-5* family.

### Changed
- requirements/setup updates; quality-of-life cleanups.

## [1.1.9b1] - 2025-10-29

### Removed
- Dropped `psycopg2` dependency in favor of `psycopg2-binary` only.

## [1.1.9b0] - 2025-10-27

### Changed
- Loosened pinned dependencies (`Jinja2`, `Flask`, `Flask-Cors`, `Flask-RESTful`)
  from `==` to `>=` in both `setup.py` and `requirements.txt` to ease coexistence
  with downstream apps that pull newer versions of these.

## [1.1.8] - 2025-07-20

### Added
- Azure OpenAI support, including configurable `host`, `port`, and `api_key`.
- `sympy` requirement for downstream features.

### Fixed
- Structural classification path when host is unset.

### Changed
- Loosened the `litellm` version constraint.
- Various dependency updates: `pglast`, `tiktoken`, `psycopg2`.

## [1.1.7a11] - 2025-03-21

### Fixed
- Bug in handling table aliases.

### Changed
- CI: removed `faiss` and `spacy` from the workflow run.

## [1.1.7a10] - 2024-09-28

### Added
- Support for `LIMIT` clauses in compiled queries.
- Allow `unprotected` mode on raw SQL passthrough as well.

## [1.1.7a9] - 2024-09-28

### Removed
- `spacy` and `FlagEmbedding` requirements (no longer needed for the default
  install; downstream apps that need them must install separately).

## [1.1.7a8] - 2024-09-28

### Changed
- Refactored dependencies; updated default GPT version.
- Escaped column and table names during embedding initialization.

## [1.1.7a7] - 2024-06-10

### Added
- Accept a list of texts as input to pure free-text queries.

### Changed
- Slight parser-prompt modifications.

## [1.1.7a6] - 2024-05-09

### Added
- Multi-join support via `_extract_recursive_joins`.

### Fixed
- De-duplicate repeated IDs in `faiss_embedding`.

## [1.1.7a5] - 2024-05-08

### Added
- 2-directional check for `opening_hours`.

## [1.1.7a4] - 2024-05-08

### Changed
- New syntax for `opening_hours`.

## [1.1.7a3] - 2024-05-07

### Fixed
- [#19](https://github.com/stanford-oval/suql/issues/19).

## [1.1.7a2] - 2024-05-07

### Fixed
- [#20](https://github.com/stanford-oval/suql/issues/20).

## [1.1.7a1] - 2024-05-02

### Fixed
- `_check_required_params` regression.

## [1.1.7a0] - 2024-05-02

### Added
- `_check_required_params` validation.

## [1.1.7a] - 2024-05-02

### Added
- `_check_required_params` exposed as an experimental feature.

## [1.1.6] - 2024-04-29

### Added
- **`faiss_embedding.py` now caches embeddings to disk by default.** Cache
  location is resolved via `platformdirs` (the user's standard cache dir).
  When `cache_embedding` is enabled (default: on), a hash of the free-text
  values is computed; on subsequent server runs, if the underlying values are
  unchanged, the cached embeddings are loaded directly. If values changed, the
  embeddings are recomputed. See the
  [`MultipleEmbeddingStore.add` API docs](https://stanford-oval.github.io/suql/suql/faiss_embedding.html#suql.faiss_embedding.MultipleEmbeddingStore.add).
- `platformdirs` added as a requirement.

## [1.1.5] - 2024-04-25

### Fixed
- [#15](https://github.com/stanford-oval/suql/issues/15).

## [1.1.4a3] - 2024-04-17

### Changed
- Logging changes.

## [1.1.4a2] - 2024-04-17

### Fixed
- Returned-results count handling.

## [1.1.4a1] - 2024-04-17

### Added
- Basic standalone `answer` support.

## [1.1.4a0] - 2024-04-16

### Added
- Internal helper `_extract_all_free_text_fcns` and `_ExtractAllFreeTextFncs`
  visitor in the SUQL compiler — enumerates every free-text function call in a
  query as `(field, query)` tuples. Foundational for the standalone `answer`
  support that landed in 1.1.4a1.

## [1.1.3] - 2024-04-15

### Changed
- Downgraded `spacy` for broader compatibility.
- Citation update; README cleanups; internal print-statement cleanup.

## [1.1.2] - 2024-04-12

### Removed
- `openai` from `setup.py` (now provided via `litellm`).

## [1.1.1] - 2024-04-11

### Changed
- **Migrated from raw OpenAI client to `litellm`.**
- Removed the engine-model map (now handled by `litellm`).
- env-management improvements.

### Added
- Logging and additional docstrings.

### Removed
- `transformers` dependency.

## [1.1.1a0] - 2024-04-08

### Added
- Python 3.8–3.11 support.

### Removed
- `torch` requirement.

## [1.1.0a0] - 2024-04-08

### Changed
- Modified `suql_execute` call structure.
- Privatized helper methods on the SUQL compiler.
- Documentation/site build pipeline (`pdoc`, `docs.yml`) overhauled.
- Moved `OpenAI()` instantiation inside the function (lazy init).

## [1.0.0b0] - 2024-04-05

### Changed
- Beta-testing version. Tested with Python 3.10 and a T4 GPU.

## [1.0.0a3] - 2024-04-05

### Added
- Prompt files included in the package.

### Removed
- `pymongo` dependency.

## [1.0.0a2] - 2024-04-04

### Added
- `__init__.py` for the `suql` package.