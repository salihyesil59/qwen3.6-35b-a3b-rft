"""Re-scores a saved eval file with the current graders, without running the model.

Truncated solutions (finish == "length") stay wrong. The result is appended to summary.csv as "<label>-rescored".

  python eval/rescore.py eval/results/base-colab-fp8_20261007-xxxxxx.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

import run_eval as R

TASKS_DIR = Path(__file__).parent / "tasks"


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("results", help="<label>_<timestamp>.jsonl written by run_eval.py or the 02 notebook")
    args = ap.parse_args()

    path = Path(args.results)
    label, stamp = path.stem.rsplit("_", 1)
    tasks = {}
    for p in TASKS_DIR.glob("*.jsonl"):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                t = json.loads(line)
                tasks[(p.stem, t["id"])] = t

    stats, changed = {}, 0
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        task = tasks.get((r["source"], r["id"]))
        if task is None:
            raise KeyError(f"Task not found: {r['source']}/{r['id']} (in {TASKS_DIR})")
        if task["type"] == "tool" and "<function=" not in r["response"]:
            ok = r["ok"]   # LM Studio returns tool_calls in a separate field, so they cannot be re-scored from the text
        else:
            ok = r.get("finish", "stop") == "stop" and R.grade(task, {"content": r["response"]})[0]
        changed += ok != r["ok"]
        s = stats.setdefault(r["source"], {"n": 0, "correct": 0, "tokens": 0, "seconds": 0.0})
        s["n"] += 1
        s["correct"] += int(ok)
        s["tokens"] += r["tokens"]
    print(f"Solutions whose score changed: {changed}")
    R.write_summary(R.RESULTS_DIR, stamp, f"{label}-rescored", "", stats)


if __name__ == "__main__":
    main()
