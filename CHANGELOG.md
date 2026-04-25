# Changelog

All notable changes to SUQL are documented in this file. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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