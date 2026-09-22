#!/usr/bin/env python3
"""LLM teacher: label multilingual knee-MRI radiology reports with tri-state findings (train only).

Uses the Anthropic Message Batches API (50% cost, async) with structured JSON output so every row is
schema-valid. Output is a JSONL consumed by scripts/make_report_labels.py (--llm-json), which calibrates
states into soft targets on the gold rows of each training fold.

  export ANTHROPIC_API_KEY=...            # or `ant auth login`
  python scripts/label_reports_llm.py submit --train-csv data/train.csv --state work/llm_batches.json
  python scripts/label_reports_llm.py collect --state work/llm_batches.json --out work/llm_labels.jsonl
  python scripts/label_reports_llm.py direct --train-csv data/train.csv --limit 20 --out work/llm_sample.jsonl

Cost guide (Opus 5, ~0.7k input + ~1k output tokens per report, batch pricing): roughly $0.02 per report.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kneemri.schema import ID_COL, TARGETS  # noqa: E402

MODEL = "claude-opus-5"
KEYS = {t: re.sub(r"[^a-z0-9]+", "_", t.lower()).strip("_") for t in TARGETS}  # "Baker's" -> "baker_s"
STATE_TO_INT = {"positive": 1, "negative": -1, "unmentioned": 0}

SCHEMA = {
    "type": "object",
    "properties": {
        "language": {"type": "string", "description": "ISO 639-1 code of the report language"},
        "findings": {
            "type": "object",
            "properties": {k: {"type": "string", "enum": ["positive", "negative", "unmentioned"]} for k in KEYS.values()},
            "required": list(KEYS.values()),
            "additionalProperties": False,
        },
        "evidence": {"type": "string", "description": "one short line quoting the decisive phrases"},
    },
    "required": ["language", "findings", "evidence"],
    "additionalProperties": False,
}

SYSTEM = """You are a musculoskeletal radiologist extracting structured findings from knee MRI reports.
Reports may be in any language (Spanish, Dutch, German, French, Portuguese, Italian, English, ...).
For each of the twelve findings return exactly one state:
  positive    - the report asserts the abnormality is present (any grade/partial/complete, unless explicitly
                described as old and fully healed with no residual abnormality)
  negative    - the report explicitly states the structure is normal/intact or the finding is absent
  unmentioned - the report does not address it
Definitions:
  acl: anterior cruciate ligament tear/rupture, partial tear, mucoid degeneration, or a torn graft.
  mcl: medial collateral ligament sprain (grade 1-3), tear, or thickening/oedema.
  medial_meniscus / lateral_meniscus: tear of any type (horizontal, radial, flap, bucket-handle, complex,
    root, degenerative tear = grade 3 signal reaching a surface), maceration, extrusion with tear, or prior
    meniscectomy with residual tear. Intrasubstance grade 1-2 signal alone is negative.
  medial_oa / lateral_oa: tibiofemoral osteoarthritis or chondral damage in that compartment (cartilage
    thinning, chondral defects/fissures, osteophytes, subchondral sclerosis/cysts, joint-space narrowing).
  pf_oa: patellofemoral osteoarthritis or chondral damage (chondromalacia patellae, trochlear/patellar
    cartilage defects, patellofemoral osteophytes).
  effusion: joint effusion beyond a trace/physiological amount.
  synovitis: synovial thickening, proliferation, hypertrophy, or inflammatory synovitis.
  baker_s: Baker's (popliteal) cyst, including ruptured.
  contusion: bone marrow oedema / bone bruise / trabecular microfracture of any bone at the knee.
  fracture: any fracture (avulsion incl. Segond, tibial plateau, osteochondral, insufficiency, stress),
    acute or subacute; a healed remote fracture without current abnormality is negative.
Rules: a laterality that is not stated does not make a meniscus/compartment finding positive for both sides;
"medial meniscus tear" leaves lateral_meniscus unmentioned unless the lateral meniscus is described.
Do not infer a finding from clinical history or the reason for the exam. Answer only with the JSON object."""


def build_params(report: str, effort: str, max_tokens: int) -> dict:
    return {
        "model": MODEL,
        "max_tokens": max_tokens,
        "system": [{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": f"<report>\n{report}\n</report>"}],
        "output_config": {"effort": effort, "format": {"type": "json_schema", "schema": SCHEMA}},
    }


def parse_message(msg) -> dict | None:
    if getattr(msg, "stop_reason", None) == "refusal":
        return None
    text = next((b.text for b in msg.content if b.type == "text"), "")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    labels = {t: STATE_TO_INT.get(data["findings"].get(k, "unmentioned"), 0) for t, k in KEYS.items()}
    return {"labels": labels, "language": data.get("language"), "evidence": data.get("evidence", "")}


def load_reports(path: str, limit: int | None) -> list[tuple[str, str]]:
    df = pd.read_csv(path)
    df[ID_COL] = df[ID_COL].astype(str)
    rows = [(sid, str(r)) for sid, r in zip(df[ID_COL], df["Report"].fillna("")) if str(r).strip()]
    return rows[:limit] if limit else rows


def cmd_submit(a):
    import anthropic

    client = anthropic.Anthropic()
    rows = load_reports(a.train_csv, a.limit)
    state = {"batches": [], "n": len(rows)}
    for i in range(0, len(rows), a.chunk):
        chunk = rows[i : i + a.chunk]
        batch = client.messages.batches.create(requests=[
            {"custom_id": f"s{j + i}", "params": build_params(rep, a.effort, a.max_tokens)} for j, (_, rep) in enumerate(chunk)
        ])
        state["batches"].append({"id": batch.id, "ids": [sid for sid, _ in chunk], "offset": i})
        print(f"submitted batch {batch.id}: {len(chunk)} requests", flush=True)
    Path(a.state).parent.mkdir(parents=True, exist_ok=True)
    Path(a.state).write_text(json.dumps(state, indent=1))


def cmd_collect(a):
    import anthropic

    client = anthropic.Anthropic()
    state = json.loads(Path(a.state).read_text())
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_ok = n_bad = 0
    with out.open("w") as f:
        for b in state["batches"]:
            while True:
                batch = client.messages.batches.retrieve(b["id"])
                if batch.processing_status == "ended":
                    break
                print(f"{b['id']}: {batch.processing_status}, processing={batch.request_counts.processing}", flush=True)
                time.sleep(a.poll)
            for result in client.messages.batches.results(b["id"]):
                idx = int(result.custom_id[1:]) - b["offset"]
                sid = b["ids"][idx]
                parsed = parse_message(result.result.message) if result.result.type == "succeeded" else None
                if parsed is None:
                    n_bad += 1
                    continue
                n_ok += 1
                f.write(json.dumps({ID_COL: sid, **parsed}, ensure_ascii=False) + "\n")
    print(f"collected {n_ok} labels ({n_bad} failed/refused) -> {out}")


def cmd_direct(a):
    """Small synchronous run (tests / spot checks). Includes server-side refusal fallbacks by default."""
    import anthropic

    client = anthropic.Anthropic()
    rows = load_reports(a.train_csv, a.limit or 20)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for sid, rep in rows:
            params = build_params(rep, a.effort, a.max_tokens)
            try:
                msg = client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **params)
            except anthropic.BadRequestError:  # fallback param unsupported on this endpoint/org: plain call
                msg = client.messages.create(**params)
            parsed = parse_message(msg)
            if parsed is None:
                print(f"{sid}: no label (refusal or unparseable)")
                continue
            f.write(json.dumps({ID_COL: sid, **parsed}, ensure_ascii=False) + "\n")
            print(sid[-12:], parsed["language"], {t: v for t, v in parsed["labels"].items() if v != 0})


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("submit", "direct"):
        p = sub.add_parser(name)
        p.add_argument("--train-csv", required=True)
        p.add_argument("--limit", type=int, default=None)
        p.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
        p.add_argument("--max-tokens", type=int, default=4096)
        if name == "submit":
            p.add_argument("--state", default="work/llm_batches.json")
            p.add_argument("--chunk", type=int, default=2000)
        else:
            p.add_argument("--out", default="work/llm_sample.jsonl")
    p = sub.add_parser("collect")
    p.add_argument("--state", default="work/llm_batches.json")
    p.add_argument("--out", default="work/llm_labels.jsonl")
    p.add_argument("--poll", type=int, default=60)
    a = ap.parse_args()
    {"submit": cmd_submit, "collect": cmd_collect, "direct": cmd_direct}[a.cmd](a)


if __name__ == "__main__":
    main()
