"""Downloads benchmark subsets and writes them to eval/tasks/ in the run_eval.py format.

Requires: pip install datasets
GPQA is gated: accept the dataset's terms on Hugging Face and log in with `hf auth login`.
Do not publish GPQA questions anywhere public.

  python eval/fetch_benchmarks.py
"""

import json
import random
import sys
from pathlib import Path

from datasets import load_dataset

TASKS_DIR = Path(__file__).parent / "tasks"
SEED = 42


def write(name, tasks):
    path = TASKS_DIR / f"{name}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"{name}: {len(tasks)} tasks -> {path}")


def math500(n=100):
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    rows = random.Random(SEED).sample(list(ds), n)
    return [{"id": r["unique_id"], "type": "math", "prompt": r["problem"], "answer": r["answer"]} for r in rows]


def aime2025():
    ds = load_dataset("MathArena/aime_2025", split="train")
    return [{"id": f"aime25-{i}", "type": "math", "prompt": r["problem"], "answer": str(r["answer"])}
            for i, r in enumerate(ds)]


def gpqa_physics():
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    rng = random.Random(SEED)
    tasks = []
    for i, r in enumerate(ds):
        if r["High-level domain"] != "Physics":
            continue
        choices = [r["Correct Answer"], r["Incorrect Answer 1"], r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
        choices = [c.strip() for c in choices]
        correct = choices[0]
        rng.shuffle(choices)
        letters = "ABCD"
        body = r["Question"].strip() + "\n\n" + "\n".join(f"{letters[j]}) {c}" for j, c in enumerate(choices))
        tasks.append({"id": f"gpqa-phys-{i}", "type": "mcq", "prompt": body,
                      "answer": letters[choices.index(correct)]})
    return tasks


def main():
    if hasattr(sys.stdout, "reconfigure"):  # for non-ASCII output on the Windows console (not on Colab)
        sys.stdout.reconfigure(encoding="utf-8")
    TASKS_DIR.mkdir(exist_ok=True)
    for name, fn in [("math500", math500), ("aime2025", aime2025), ("gpqa_physics", gpqa_physics)]:
        try:
            write(name, fn())
        except Exception as e:  # if one dataset fails, still download the others
            print(f"{name}: DOWNLOAD FAILED -> {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
