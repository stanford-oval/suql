# One-time setup: fixtures repo + PAT + secrets

This walkthrough wires up the auth-gated fetch the CI workflow depends on.
Do it once; CI handles itself from there.

Notation: `<owner>` is your GitHub username/org. `<fixtures-repo>` is the
name you'll pick for the data repo (e.g. `suql-test-fixtures`).

## Step 1 — Create the private fixtures repo

The sample CSV needs a home separate from the SUQL code. Create a new
**private** repo:

```bash
gh repo create <owner>/<fixtures-repo> --private --description \
    "Sample data + fixtures consumed by SUQL CI"
```

That's it for now — it can stay empty until you push the first sample.

## Step 2 — Drop in the first sample

Push a CSV named (by default) `acled_sample.csv` matching the column
order in `tests/fixtures/schema.sql`:

```bash
gh repo clone <owner>/<fixtures-repo> /tmp/fixtures
cp /path/to/your/sample.csv /tmp/fixtures/acled_sample.csv
cd /tmp/fixtures
git add acled_sample.csv
git commit -m "Initial sample"
git push
```

For a different filename, set the `SAMPLE_CSV` env in the workflow's
"Ingest sample data" step to match.

## Step 3 — Create a fine-grained PAT

GitHub's fine-grained PATs let you scope a token to a single repo with
read-only permission. That's exactly what CI needs.

1. Open: **GitHub → Settings → Developer settings → Personal access tokens
   → Fine-grained tokens → Generate new token**
2. **Token name:** something memorable, e.g. `suql-ci-fixtures-readonly`
3. **Expiration:** pick the longest you're comfortable with (max 1 year).
   Set a calendar reminder to rotate before expiry.
4. **Repository access:** *Only select repositories* → `<owner>/<fixtures-repo>`
5. **Permissions → Repository permissions:**
   - `Contents`: **Read-only**
   - `Metadata`: **Read-only** (required by GitHub for any access)
6. Click **Generate token**, copy the value — GitHub only shows it once.

## Step 4 — Add secrets + variable to the SUQL repo

1. Open: **SUQL fork → Settings → Secrets and variables → Actions →
   Secrets tab → New repository secret**
2. Add three secrets:

| Name              | Value                                          |
| ----------------- | ---------------------------------------------- |
| `FIXTURES_TOKEN`  | the PAT from Step 3                            |
| `OPENAI_API_KEY`  | your LLM proxy key                             |
| `OPENAI_API_BASE` | your LLM proxy base URL                        |

3. Switch to the **Variables** tab and add one:

| Name             | Value                              |
| ---------------- | ---------------------------------- |
| `FIXTURES_REPO`  | `<owner>/<fixtures-repo>`          |

## Step 5 — Verify by triggering a CI run

```bash
git push
# or
gh workflow run "SUQL liveness check"
```

Watch: **repo → Actions → SUQL liveness check → most-recent run**.

The "Fetch sample CSV" step is where the PAT is exercised. If it fails:
- 404 → PAT can see fewer repos than expected; recheck Step 3's
  *Repository access* selection.
- 403 → PAT lacks `Contents: Read`; recheck Step 3's *Permissions*.
- Anything else → check the run logs.

The "Run liveness probe" step is the actual end-to-end check. On
success it prints `✓ suql_execute returned in Ns` with row count and
cost.

## Rotation

When the PAT approaches expiration (GitHub emails you ahead of time):

1. Generate a new PAT with the same scope (Step 3)
2. Update the `FIXTURES_TOKEN` secret with the new value (Step 4)
3. Delete the old PAT

## Revocation (if the PAT leaks)

1. **Immediately** delete the leaked PAT at GitHub → Settings → Developer
   settings → Personal access tokens → Fine-grained tokens
2. Generate a new PAT, update the secret, push a no-op commit to verify
3. Review whatever process exposed the PAT
