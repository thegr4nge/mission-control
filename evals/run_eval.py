"""Run the mission eval on one model and save transcripts and scores.

  pixi run eval --model claude                 # Claude via the Claude Code CLI on your plan
  pixi run eval --model gemini                 # needs GEMINI_API_KEY in .env
  pixi run eval --model groq --tasks T01,T05   # needs GROQ_API_KEY in .env
  pixi run eval --model gemini --list-models   # what this key can call
  pixi run eval --rescore evals/results/x.json # re-score a saved run after changing scoring
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path

import anyio

from agents.model_adapter import ChatModel, ModelError
from evals.runners import run_claude_task, run_openai_tasks
from evals.scoring import Score, ToolCall, Transcript, score
from evals.tasks import TASKS

RESULTS = Path(__file__).resolve().parent / "results"


def pick_tasks(spec: str | None):
    if not spec:
        return list(TASKS)
    wanted = [s.strip() for s in spec.split(",") if s.strip()]
    chosen = [t for t in TASKS if any(t.id == w or t.id.startswith(w + "_") for w in wanted)]
    if len(chosen) != len(wanted):
        sys.exit(f"unknown task in {spec!r}; ids: {', '.join(t.id for t in TASKS)}")
    return chosen


def summary_table(scores: list[Score]) -> str:
    rows = ["task                         tools  errs  correct  grounded  explain",
            "---------------------------  -----  ----  -------  --------  -------"]
    for s in scores:
        flag = f"  ERROR: {s.error[:70]}" if s.error else ""
        rows.append(f"{s.task_id:<27}  {s.tool_use:5.2f}  {s.tool_errors:4d}  {s.correctness:7.2f}  "
                    f"{s.grounding:8.2f}  {s.explanation:7.2f}{flag}")
    n = len(scores) or 1
    rows.append(f"{'MEAN':<27}  {sum(s.tool_use for s in scores) / n:5.2f}  "
                f"{sum(s.tool_errors for s in scores):4d}  {sum(s.correctness for s in scores) / n:7.2f}  "
                f"{sum(s.grounding for s in scores) / n:8.2f}  {sum(s.explanation for s in scores) / n:7.2f}")
    return "\n".join(rows)


def result_path(label: str) -> Path:
    RESULTS.mkdir(exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9.-]+", "_", label)
    return RESULTS / f"{safe}-{time.strftime('%Y%m%d-%H%M%S')}.json"


def save(label: str, transcripts: list[Transcript], scores: list[Score], path: Path) -> Path:
    """Full transcripts as JSON (gitignored: they carry local paths), and the score table as markdown
    beside it (committed)."""
    path.write_text(json.dumps({"model": label, "scores": [s.to_dict() for s in scores],
                                "transcripts": [asdict(t) for t in transcripts]}, indent=2), encoding="utf-8")
    md = path.with_suffix(".md")
    md.write_text(f"# {label}\n\n{time.strftime('%Y-%m-%d %H:%M')}, {len(scores)} tasks\n\n```\n"
                  f"{summary_table(scores)}\n```\n", encoding="utf-8")
    return path


def load_transcripts(path: Path) -> list[Transcript]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for t in data["transcripts"]:
        t["tool_calls"] = [ToolCall(**c) for c in t["tool_calls"]]
        out.append(Transcript(**t))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["claude", "gemini", "groq", "ollama"])
    ap.add_argument("--model-name", help="override the provider's default model")
    ap.add_argument("--tasks", help="comma-separated task ids or prefixes, e.g. T01,T05")
    ap.add_argument("--list-models", action="store_true")
    ap.add_argument("--rescore", type=Path)
    a = ap.parse_args(argv)

    by_id = {t.id: t for t in TASKS}
    if a.rescore:
        transcripts = load_transcripts(a.rescore)
        scores = [score(by_id[t.task_id], t) for t in transcripts]
        print(summary_table(scores))
        print(f"rewrote {save(transcripts[0].model if transcripts else 'empty', transcripts, scores, a.rescore)}")
        return 0
    if not a.model:
        ap.error("--model is required")
    tasks = pick_tasks(a.tasks)

    done: list[Transcript] = []
    path = result_path(a.model if not a.model_name else f"{a.model}_{a.model_name}")

    def progress(tr: Transcript):
        s = score(by_id[tr.task_id], tr)
        status = f"ERROR {tr.error[:80]}" if tr.error else f"correct {s.correctness:.2f}"
        print(f"  {tr.task_id}: {len(tr.tool_calls)} tool calls, {tr.seconds:.0f} s, {status}", flush=True)
        done.append(tr)  # save after every task so a crash or rate limit loses nothing
        save(tr.model, done, [score(by_id[t.task_id], t) for t in done], path)

    if a.model == "claude":
        print(f"Running {len(tasks)} tasks on Claude via the Claude Code CLI", flush=True)
        for t in tasks:
            progress(run_claude_task(t, a.model_name))
        transcripts = done
    else:
        try:
            m = ChatModel(a.model, a.model_name)
            if a.list_models:
                print("\n".join(m.list_models()))
                return 0
        except ModelError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(f"Running {len(tasks)} tasks on {m.label}", flush=True)
        transcripts = anyio.run(lambda: run_openai_tasks(m, tasks, on_done=progress))

    scores = [score(by_id[t.task_id], t) for t in transcripts]
    label = transcripts[0].model if transcripts else a.model
    print(summary_table(scores))
    print(f"saved {save(label, transcripts, scores, path)} and {path.with_suffix('.md').name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
