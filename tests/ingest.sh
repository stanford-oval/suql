#!/usr/bin/env bash
# Apply the schema and \COPY a sample CSV into Postgres.
#
# This script is auth-agnostic: it expects the CSV to already exist on disk at
# $SAMPLE_CSV. Fetching the CSV (from a private fixtures repo, an S3 bucket,
# wherever) is the *caller's* job — keeps secrets handling out of this script.
#
# In CI, the workflow checks out the fixtures repo first and points SAMPLE_CSV
# at the resulting file. Locally, you drop a CSV anywhere and export the path.
#
# Env vars:
#     SAMPLE_CSV     (required) absolute path to the CSV to load
#     PGHOST         default 127.0.0.1
#     PGPORT         default 5432
#     PGDATABASE     default acled
#     PGUSER         default oval
#     PGPASSWORD     (required, passed via env not flag)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCHEMA="$HERE/fixtures/schema.sql"

: "${SAMPLE_CSV:?SAMPLE_CSV must point at a CSV file on disk}"
: "${PGHOST:=127.0.0.1}"
: "${PGPORT:=5432}"
: "${PGDATABASE:=acled}"
: "${PGUSER:=oval}"
: "${PGPASSWORD:?PGPASSWORD must be set}"

export PGPASSWORD

[ -r "$SCHEMA" ]     || { echo "✗ schema not found: $SCHEMA" >&2; exit 1; }
[ -r "$SAMPLE_CSV" ] || { echo "✗ sample CSV not found: $SAMPLE_CSV" >&2; exit 1; }

ROW_COUNT="$(wc -l < "$SAMPLE_CSV" | tr -d ' ')"
echo "→ Sample CSV: $SAMPLE_CSV ($ROW_COUNT lines incl. header)"

echo "→ Applying schema..."
psql -v ON_ERROR_STOP=1 -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDATABASE" \
    -f "$SCHEMA" > /dev/null

echo "→ Loading CSV..."
psql -v ON_ERROR_STOP=1 -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDATABASE" \
    -c "\\COPY events FROM '$SAMPLE_CSV' WITH (FORMAT csv, HEADER true)"

INGESTED="$(psql -t -A -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDATABASE" \
    -c "SELECT COUNT(*) FROM events;")"
echo "✓ events table contains $INGESTED rows"
