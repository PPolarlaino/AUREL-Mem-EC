# AUREL-Mem-EC

Version: `v3-ec-20261030`. An independent deployment of the AUREL-Mem v3 candidate
with complete-record evidence filling. It exposes synchronous Add/Search only;
the leaderboard owns Answer and Eval. No GPU or model training is required.

This is a release candidate, not an official leaderboard result. Endpoint
review, Eval Key issuance and official Smoke/Full are still pending.
Repository: https://github.com/PPolarlaino/AUREL-Mem-EC

## Provenance and result

Based on [Crosser-XDU/AUREL-Mem](https://github.com/Crosser-XDU/AUREL-Mem), fixed
upstream `c3f9ff4b71b09117dac446fd474ce96c87481e65`, with local v3 changes at
`ebb4d20cdc6dbdcf43532ab74256c0bc03c66ac7`. See `source-lock.json` for the exact
vendored source hashes and `docs/upstream-research-disclosure.md` for upstream
research attribution. The original repository is not modified by this package.

The Brand development source's29 open questions scored37.93% for controlled
original AUREL,31.03% for v3, and44.83% for v3 with evidence filling. These are
local GPT-4o-mini proxy scores, not full MemEye, three-dataset, or official scores.
No benchmark messages, images, predictions, gold answers or credentials are
included in this repository.

## What is frozen

- The candidate's native extraction/retrieval/storage code is copied unchanged.
- The experiment's shared source-slot extraction contract and complete-record
  recovery are retained, including strict embedding response validation.
- The exact evidence reorder prototype and local sizing implementation are
  retained. They run CPU-only with an in-memory dummy response; no final Answer
  or judge model is called by this service.
- One generic delivery profile is used for every request. It never inspects
  benchmark labels in user IDs. Up to10 candidates are retrieved, even when AML
  supplies its formal upper bound `top_k=100`; a smaller requested bound is obeyed.
- Full records and original image bytes are preserved. Records exceeding the
  response's cumulative30 MiB image cap are omitted whole, never partially
  rewritten. This transport limit was not an experimental ablation.
- Optional old relevance `score` fields are omitted on the wire after budget
  ordering. This leaves the local Answer input unchanged and avoids publishing
  scores which do not describe the returned delivery order.

AML documents a117,760-token Answer input budget and rank-prefix selection.
Our sizing uses the frozen local conservative estimator, not a claimed exact
copy of the platform's private packing code. Official gains remain to be tested.

## Run

Python3.12; install `requirements.lock.txt` in a dedicated environment. Place
private `.env.local` and `.env.leaderboard.local` in a separate config directory.
The former supplies existing `OPENAI_*` generation/embedding settings; the latter
must contain a random dedicated `MEMORY_API_TOKEN` of at least32 characters.
`AML_EVAL_KEY` belongs to the leaderboard client and is never sent to the service
or model providers. The Qwen evaluator configuration is not used.

```sh
export AUREL_EC_CONFIG_ROOT=/absolute/private/config
export AUREL_EC_STATE=/absolute/private/state
export AUREL_EC_BUDGET_POLICY=/absolute/private/approved-budget.json
python -m uvicorn aurel_ec.service:app_factory --factory --host 127.0.0.1 --port 18086 --workers 1 --no-access-log
```

The bundled `deploy/budget.disabled.json` authorizes zero model calls. With it,
the service can start, but `/healthz` and model operations return503. A concrete
approved budget is required before presenting the endpoint as ready for Smoke.
Old experiment ledgers and unused historical balances are never reused.

Add: `POST /add`; Search: `POST /search`; public health: `GET /healthz`;
public version: `GET /version`. Send `Authorization: Token <MEMORY_API_TOKEN>`
or Bearer for Add/Search. All API errors omit source text and credentials.

Run one worker and publish behind a stable HTTPS reverse proxy/tunnel. Start
with platform Add/Search concurrency1. Upstream transport is serial,120 seconds
per attempt, with no automatic network retries; native schema correction remains
bounded and every HTTP attempt is counted. Platform retries can incur new calls;
the same successful Add request is idempotent. Model request prices are external
and this request counter is not a currency estimate.

Source memories are private data in the state directory. Raw model-response and
HTTP access logs are disabled; the request ledger retains metadata only. Apply
the platform's required data-retention/deletion policy after the evaluation.
Full-scale capacity and official source distributions have not been validated.

## Offline verification

```sh
python -m pytest -q tests
```

Tests use an injected HTTP transport with synthetic responses. They check the
real native extraction, database, retrieval and Add/Search routes without an
external model request. Passing them is not the official Smoke test.

## Publication notice

On2026-10-30, the submitting project owner confirmed that the collaborator
responsible for AUREL-Mem permits public release of this modified version.
The upstream snapshot has no top-level LICENSE file. This publication permission
is not represented as a blanket license for third-party reuse, and this package
does not relicense upstream code. Dependency licenses remain their owners'.

Official contract: https://agentmemoryleaderboard.ai/api-guide
