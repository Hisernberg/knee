# red_tape_raccoon — autonomous healthcare-administration agent

Codename for our entry in the Kaggle hackathon **`chi-bench`** (IEEE Big Data Cup 2026, χ-Bench
Healthcare AI Agents Challenge). An agent works long-horizon U.S. healthcare-administration cases
inside the χ-World simulator. There are three tracks: provider prior authorization, payer
utilization management, and care management, with 25 public tasks each. Each task is scored
strictly, as 0 or 1 (every verifier check must pass). Final prizes are decided by an
organizer-run evaluation on unreleased tasks: pass@1 first, then pass^3, then inference cost,
then tool calls.

## Status (2026-09-23)

| Item | State |
|---|---|
| Official repo + public tasks (`chi-bench-v1.0.0`) | downloaded; pinned to chi-bench `2395e9e` |
| Managed-Care Operations Handbook (gated) | **access not yet approved** for the team's HF account; the rules forbid running trials before approval |
| Docker | dependency stage `chi-bench:ci` built locally; runtime stage bakes the handbook in, so it waits on approval |
| Kaggle submission builder | `scripts/make_kaggle_submission.py`, verified to reproduce a published packet's pass@1 (0.5467) from its verifier files |
| Agent harness | in design (research in progress) |
| API keys | `ANTHROPIC_API_KEY` needed for the official judge (and for Claude agents); not yet provided |

## Layout

```
scripts/setup_env.sh              reproducible env: clone chi-bench @ pinned commit, deps, tasks, handbook, image
scripts/local_dockerfile.py       Dockerfile variant for TLS-intercepting build hosts (sandbox only)
scripts/make_kaggle_submission.py submission.csv strictly from verifier/reward.json files
raccoon/                          the agent harness (Harbor agent class + prompts + playbooks)
configs/                          chi-bench submission configs
docs/                             analysis, plan, runbook, write-ups
report/                           IEEE qualifying report (<= 5 pages)
vendor/                           (git-ignored) chi-bench clone, data, handbook, run logs
```

## Rules we must not break

- Never commit or upload the handbook, excerpts of it, `.env`, or any token.
- A Kaggle reward may only come from an official `verifier/reward.json` (one run per task).
- No human intervention once a trial starts; no task-specific hard-coded answers.
- Run every `cb` command with `ANTHROPIC_BASE_URL` unset on hosts that set it for other tools:
  Harbor forwards that variable into the trial container.
