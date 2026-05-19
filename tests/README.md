# SUQL liveness pipeline

End-to-end CI check: bring up Postgres, ingest a private data sample,
start SUQL's two servers, run one canned `answer()` query, confirm it
returns rows. Runs on every push via `.github/workflows/test.yml`.

The answer to "did this change break anything obvious?" is a green or
red dot on the commit — no local environment needed to find out.

## Layout

```
tests/
├── README.md                       this file
├── SETUP.md                        one-time fixtures repo + PAT setup
├── check_alive.py                  liveness probe — runs the canned query
├── ingest.sh                       applies schema + \COPY (auth-agnostic)
├── start_embedding_server.py       embedding-server bootstrap
└── fixtures/
    └── schema.sql                  example events table + select_user / creator_role
```

No CSV ever lives in this repo. The data sits in a separate private
fixtures repo that CI fetches with a scoped credential. See `SETUP.md`.

## Required GitHub secrets

| Secret             | What it is                                                 |
| ------------------ | ---------------------------------------------------------- |
| `FIXTURES_TOKEN`   | fine-grained PAT, `Contents: Read` on the fixtures repo    |
| `OPENAI_API_KEY`   | LLM proxy credential                                       |
| `OPENAI_API_BASE`  | proxy base URL                                             |

## Optional GitHub variable

| Variable          | Default                       | What it is                       |
| ----------------- | ----------------------------- | -------------------------------- |
| `FIXTURES_REPO`   | `skyxiath/suql-test-fixtures` | `owner/name` of the fixtures repo |

Set these at: **repo → Settings → Secrets and variables → Actions**.

See `SETUP.md` for the walkthrough.

## Adapting the probe to your data

The example schema + canned query assume an ACLED-shaped events table.
For a different dataset:

1. Replace `fixtures/schema.sql` with your table DDL. Keep the
   `select_user` and `creator_role` blocks — SUQL's compiler needs them.
2. Push a CSV matching that schema to the fixtures repo.
3. Update `start_embedding_server.py`'s call to `store.add(...)` to
   point at your table / primary key / free-text column. Or override
   via `SUQL_TABLE`, `SUQL_ID_COL`, `SUQL_TEXT_COL` env vars.
4. Override the probe via `SUQL_QUERY` in the workflow's "Run liveness
   probe" step. Any query that exercises one `answer()` clause and
   returns rows from your sample works.

## Refreshing the sample

Sample lives in the fixtures repo, not here. Update it there:

```bash
cd /path/to/your-fixtures-repo
# (generate / copy in a new sample.csv however you do)
git add sample.csv && git commit -m "Refresh sample" && git push
```

CI on the next workflow run picks up the new sample automatically — no
change to this repo needed.

## Running locally

```bash
# 1. Start Postgres locally:
docker run -d --name suql-test-pg -p 5432:5432 \
    -e POSTGRES_USER=oval -e POSTGRES_PASSWORD=oval -e POSTGRES_DB=acled \
    postgres:16

# 2. Install SUQL with embedding deps:
pip install -e .[embedding]
pip install python-dotenv

# 3. Drop a sample CSV at /tmp/sample.csv (from wherever you keep it).

# 4. Ingest:
PGPASSWORD=oval SAMPLE_CSV=/tmp/sample.csv ./tests/ingest.sh

# 5. Start servers:
python -m suql.free_text_fcns_server &
SUQL_EMBED_PORT=8505 python tests/start_embedding_server.py &

# 6. Run probe:
OPENAI_API_KEY=... OPENAI_API_BASE=... python tests/check_alive.py
```

## What "liveness" means here

The probe runs one canned `answer(...) = 'yes'` query. Exit codes:

| Code | Meaning                                                     |
| ---- | ----------------------------------------------------------- |
| 0    | `suql_execute` returned >= 1 row                            |
| 1    | `suql_execute` raised (PG, embedding, free-text, or LLM)    |
| 2    | Ran cleanly but returned 0 rows                             |
| 3    | Preflight failed — TCP target unreachable or creds missing  |

The workflow fails on any non-zero exit.

## Scope

This catches *plumbing* breaks (server boot failure, SQL compilation,
prompt loading, network connectivity). It does not measure result
quality — for that you want benchmarks / evals, not a CI smoke test.
