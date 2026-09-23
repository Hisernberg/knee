"""Build the Kaggle `submission.csv` strictly from official verifier outputs.

Every reward comes from a trial's `verifier/reward.json` (written by the official χ-Bench
verifier); nothing is typed by hand. Competition rules (section 9): exactly one row per
required public task, reward in {0, 1}, one run per task. When a task was run more than
once, the EARLIEST trial is used (that is the pass@1 attempt) and later ones are ignored.

    python make_kaggle_submission.py --runs <trial roots...> --tasks data/tasks.jsonl \
        --out submission.csv [--sample sample_submission.csv] [--missing-as-failed]

A missing trial is an error by default. `--missing-as-failed` writes 0 for it, which matches
how the organizers score a trial that never produced a result (a failed trial).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

CORE_FAMILIES = ("care_management", "prior_auth_provider", "prior_auth_um")


def required_tasks(tasks_jsonl: Path | None, sample_csv: Path | None) -> list[str]:
    if sample_csv is not None:
        with sample_csv.open() as f:
            return [row["task_id"] for row in csv.DictReader(f)]
    rows = [json.loads(line) for line in tasks_jsonl.open() if line.strip()]
    return [r["task_id"] for r in rows if r["family"] in CORE_FAMILIES]


def collect_trials(roots: list[Path]) -> dict[str, list[dict]]:
    """task_id -> trials (each with reward, started_at, path) found under the roots."""
    found: dict[str, list[dict]] = {}
    for root in roots:
        for result_path in root.rglob("result.json"):
            reward_path = result_path.parent / "verifier" / "reward.json"
            if not reward_path.exists():
                continue
            result = json.loads(result_path.read_text())
            task_name = result.get("task_name") or ""
            task_id = task_name.split("/", 1)[-1]
            reward = json.loads(reward_path.read_text()).get("reward")
            if reward not in (0, 1, 0.0, 1.0):
                raise SystemExit(f"non-binary reward {reward!r} in {reward_path}")
            found.setdefault(task_id, []).append(
                {
                    "reward": int(reward),
                    "started_at": result.get("started_at") or "",
                    "path": str(result_path.parent),
                }
            )
    return found


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", type=Path, required=True)
    ap.add_argument("--tasks", type=Path, help="data/tasks.jsonl (defines the 75 core tasks)")
    ap.add_argument("--sample", type=Path, help="Kaggle sample_submission.csv (takes precedence)")
    ap.add_argument("--out", type=Path, default=Path("submission.csv"))
    ap.add_argument("--missing-as-failed", action="store_true")
    args = ap.parse_args()
    if args.tasks is None and args.sample is None:
        ap.error("need --tasks or --sample")

    required = required_tasks(args.tasks, args.sample)
    if len(set(required)) != len(required):
        raise SystemExit("required task list has duplicates")
    trials = collect_trials(args.runs)

    rows, missing, repeated = [], [], []
    for task_id in required:
        ts = sorted(trials.get(task_id, []), key=lambda t: t["started_at"])
        if not ts:
            missing.append(task_id)
            rows.append((task_id, 0))
            continue
        if len(ts) > 1:
            repeated.append(task_id)
        rows.append((task_id, ts[0]["reward"]))

    extra = sorted(set(trials) - set(required))
    if missing and not args.missing_as_failed:
        raise SystemExit(f"{len(missing)} required tasks have no verifier result: {missing[:10]}")
    with args.out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task_id", "reward"])
        w.writerows(rows)

    passed = sum(r for _, r in rows)
    print(f"wrote {args.out}: {len(rows)} rows, pass@1 = {passed}/{len(rows)} = {passed / len(rows):.4f}")
    if missing:
        print(f"  missing (scored 0): {len(missing)}", file=sys.stderr)
    if repeated:
        print(f"  tasks with >1 trial (earliest used): {len(repeated)}", file=sys.stderr)
    if extra:
        print(f"  ignored non-required trials: {len(extra)}", file=sys.stderr)


if __name__ == "__main__":
    main()
