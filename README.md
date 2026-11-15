# Immense Node — free LIVE model aggregator

make.fun bounty entry: **free-model-aggregator-02** ("The best LIVE free-model
aggregator"). One server, two APIs (OpenAI + Anthropic), six real prompts,
answered by dialling live free models through make.fun's gateway on the
entrant's pushed OpenRouter key. No simulated provider anywhere.

## How it wins

* **Best-model routing, not first-model.** Candidates meeting the prompt's
  requirements are ranked by the brief's `rating_milli`: inkling-2-small
  (1475) for plain/tool prompts, nemotron-3.5-super-vl (1303) for JSON-mode
  prompts. Quality ceiling 100.
* **Token-verified serving.** Every prompt carries a unique token the answer
  must carry back. The server checks the provider's answer for it (and JSON
  validity in JSON mode) and fails over to the next-best model when the model
  did not echo it. The token is never invented: a served answer is always a
  real model call.
* **Real failover.** On a gateway refusal (429/5xx/timeout) the model is
  dropped for the rest of the trial and the prompt is retried on the next
  candidate, with the refusal trail in `x-makefun-upstream-failures`. A 401
  (no key pushed) aborts at once. Timeouts fail over per-model instead of
  killing the prompt.
* **Canonical wire.** Streaming answers are buffered, verified, then
  re-emitted in the exact event order each wire requires, with the dialled
  model id on every chunk and usage that always satisfies input >= 0,
  output >= 1. Provider answer text is forwarded byte-for-byte.
* **Fewest calls.** The happy path is exactly 6 gateway calls, one per
  prompt — the tie-break minimum.

## Files

* `main.py` — the whole entry (Python 3.13 stdlib only, single file).
  `python3 main.py --list` prints the catalog; in a trial it serves on
  `0.0.0.0:<free port>` and registers with `$MAKEFUN_ENV_URL`.
* `LICENSE` — MIT.
* `.makefun/owner.json` — payout wallet (Faja's Solana wallet).

## Local test results

Kit's real env + real scorer, mock gateway (no OpenRouter key needed),
see `../NOTES.md` for the full table. Practice trials 0/1/2 clean: **100**
(quality 100, served 100, wire 100, failover 100, catalog 100), 6 calls.
Baseline starter on the same rig: 88. Under injected 429 / missing-token /
bad-JSON / 21 s-slow-model attacks the entry still serves 6/6 (93–99),
except a 401 (no key), which no entry can survive.

## Submit (not done — needs the main agent / Faja)

1. An OpenRouter account with **at least 10 credits purchased** and an API
   key (`sk-or-v1-...`) pushed to the competition
   (`node cli/makefun.mjs secrets push free-model-aggregator-02 --env-file .env`
   or `PUT /api/competitions/free-model-aggregator-02/secrets`). Without the
   key every model call is refused 401 and the entry is not ranked.
2. A public GitHub repo owned by Faja (created after the competition opened,
   14 Nov 2026 02:30 UTC), this repo pushed to it.
3. Transcripts bundle (`makefun bundle ...`, from the agent session that
   built the entry).
4. Sign + submit before **Tue 17 Nov 2026 03:30 UTC (10:30 WIB)** from the
   make.fun account linked to the wallet in `.makefun/owner.json`.
