"""Runs task sets against an LM Studio (OpenAI-compatible) server and grades the answers.

Task files are JSONL, one task per line. Supported types:
  math    : {"id", "type": "math", "prompt", "answer"}           -> the last \\boxed{} answer is compared
  numeric : {"id", "type": "numeric", "prompt", "answer": "4.70 m", "rel_tol": 0.02}
                                                                  -> numeric value within a relative tolerance (units ignored)
  mcq     : {"id", "type": "mcq", "prompt", "answer": "B"}       -> the "Answer: X" line is compared
  python  : {"id", "type": "python", "prompt", "tests"}          -> the last ```python block + asserts are run
  pytest  : {"id", "type": "pytest", "prompt", "tests"}          -> code is saved as solution.py, tests run with pytest
  tool    : {"id", "type": "tool", "prompt", "tools", "expected_call": {"name", "arguments"}}

The grading functions (build_messages, grade) are also imported by the Colab notebooks.

Example:
  python eval/run_eval.py --tasks eval/tasks/personal.jsonl --label base
"""

import argparse
import csv
import json
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import requests

RESULTS_DIR = Path(__file__).parent / "results"

MATH_SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
NUMERIC_SUFFIX = "\n\nPlease reason step by step, and put your final numerical answer (with units) within \\boxed{}."
MCQ_SUFFIX = "\n\nThink step by step, then finish with a line of the form 'Answer: X' where X is the letter."
PYTHON_SUFFIX = "\n\nReturn the complete solution in a single ```python code block."
PYTEST_SUFFIX = "\n\nReturn the complete solution in a single ```python code block. It will be saved as solution.py."

SUFFIXES = {"math": MATH_SUFFIX, "numeric": NUMERIC_SUFFIX, "mcq": MCQ_SUFFIX,
            "python": PYTHON_SUFFIX, "pytest": PYTEST_SUFFIX}


# ---------- model call ----------

def chat(args, messages, tools=None):
    body = {
        "model": args.model,
        "messages": messages,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "seed": args.seed,
    }
    if tools:
        body["tools"] = tools
    resp = requests.post(f"{args.base_url}/chat/completions", json=body, timeout=args.timeout)
    resp.raise_for_status()
    data = resp.json()
    message = data["choices"][0]["message"]
    tokens = data.get("usage", {}).get("completion_tokens", 0)
    return message, tokens


def strip_thinking(text):
    # In raw output (e.g. vLLM) the opening <think> tag stays in the prompt; only </think> is visible
    text = text or ""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return text.strip()


def build_messages(task, system=""):
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": task["prompt"] + SUFFIXES.get(task["type"], "")})
    return messages


# ---------- grading ----------

def last_boxed(text):
    idx = text.rfind("\\boxed{")
    if idx < 0:
        return None
    i, depth = idx + len("\\boxed{"), 1
    start = i
    while i < len(text) and depth:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start:i - 1] if depth == 0 else None


ORDINAL = re.compile(r"\^\{?\\(?:text|mathrm)\{(?:st|nd|rd|th)\}\}?")   # 12^{\text{th}} -> 12


def normalize_math(s):
    s = s.strip().strip("$").rstrip(".")
    s = ORDINAL.sub("", s)
    for old, new in [("\\left", ""), ("\\right", ""), ("\\!", ""), ("\\,", ""), ("\\ ", ""),
                     ("dfrac", "frac"), ("tfrac", "frac"), ("^\\circ", ""), ("^{\\circ}", ""),
                     ("\\%", ""), ("\\text{", "{"), (" ", "")]:
        s = s.replace(old, new)
    # Shorthands: \frac59 -> \frac{5}{9}, \frac{5}9 -> \frac{5}{9}, \sqrt2 -> \sqrt{2}
    s = re.sub(r"\\frac(\w)(\w)", r"\\frac{\1}{\2}", s)
    s = re.sub(r"\\frac\{([^{}]+)\}(\w)", r"\\frac{\1}{\2}", s)
    s = re.sub(r"\\frac(\w)\{", r"\\frac{\1}{", s)
    s = re.sub(r"\\sqrt(\w)", r"\\sqrt{\1}", s)
    return s


def without_units(s):
    """'5.4 \\text{ cents}' -> '5.4', '12 grade' -> '12' (only as an extra comparison candidate)."""
    s = re.sub(r"\\(?:text|mbox|mathrm)\{[^{}]*\}", "", ORDINAL.sub("", s))
    return re.sub(r"(?<=[\d}])\s*[a-zA-Z]{3,}$", "", s.strip()).strip()


def to_number(s):
    s = s.replace(",", "")
    m = re.fullmatch(r"(-?)\\frac\{(-?[\d.]+)\}\{(-?[\d.]+)\}", s) or re.fullmatch(r"(-?)([\d.]+)/([\d.]+)", s)
    if m:
        sign = -1 if m.group(1) else 1
        return sign * float(m.group(2)) / float(m.group(3))
    return float(s)


def math_equal(pred, gold):
    if pred is None:
        return False
    preds = {normalize_math(pred), normalize_math(without_units(pred))} - {""}
    golds = {normalize_math(str(gold)), normalize_math(without_units(str(gold)))} - {""}
    for p in preds:
        for g in golds:
            if p == g:
                return True
            try:
                if abs(to_number(p) - to_number(g)) < 1e-6:
                    return True
            except (ValueError, ZeroDivisionError):
                pass
    return False


def leading_number(s):
    """'4.70 m', '3.7036e9 m', '1.2 \\times 10^{-3} kg', '-34.73 degrees' -> float (unit ignored)."""
    s = s.replace(",", "").replace("\\,", "").replace("−", "-").replace("$", "")
    s = re.sub(r"\s*(\\times|\\cdot|×|x)\s*10\^\{?(-?\d+)\}?", r"e\2", s)
    m = re.match(r"\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", s)
    return float(m.group(1)) if m else None


def grade_numeric(task, content):
    pred = last_boxed(content)
    if pred is None:
        return False, None
    if math_equal(pred, task["answer"]):
        return True, pred
    p, g = leading_number(pred), leading_number(str(task["answer"]))
    if p is None or g is None:
        return False, pred
    tol = task.get("rel_tol", 0.02)
    return abs(p - g) <= tol * max(abs(g), 1e-12), pred


def grade_math(task, content):
    pred = last_boxed(content)
    return math_equal(pred, task["answer"]), pred


def grade_mcq(task, content):
    found = re.findall(r"Answer:\s*\(?([A-J])\b", content)
    pred = found[-1] if found else last_boxed(content)
    return pred is not None and pred.strip() == task["answer"], pred


def last_python_block(content):
    blocks = re.findall(r"```python\s*\n(.*?)```", content, flags=re.S)
    return blocks[-1] if blocks else None


def run_python(files, cmd, timeout):
    # Warning: this is not a real sandbox; model code runs on this machine.
    with tempfile.TemporaryDirectory() as tmp:
        for name, text in files.items():
            (Path(tmp) / name).write_text(text, encoding="utf-8")
        try:
            proc = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, "timeout"
    return proc.returncode == 0, (proc.stdout + proc.stderr)[-500:]


def grade_python(task, content, timeout):
    code = last_python_block(content)
    if code is None:
        return False, "no code block"
    return run_python({"solution.py": code + "\n\n" + task["tests"] + "\n"},
                      [sys.executable, "solution.py"], timeout)


def grade_pytest(task, content, timeout):
    code = last_python_block(content)
    if code is None:
        return False, "no code block"
    return run_python({"solution.py": code, "test_solution.py": task["tests"]},
                      [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "test_solution.py"],
                      timeout)


def parse_xml_tool_calls(text):
    """Qwen3.5/3.6 format: <tool_call><function=name><parameter=p>value</parameter></function></tool_call>"""
    calls = []
    for name, body in re.findall(r"<function=([^>\s]+)>(.*?)</function>", text or "", flags=re.S):
        args = {}
        for key, value in re.findall(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", body, flags=re.S):
            try:
                args[key] = json.loads(value)
            except json.JSONDecodeError:
                args[key] = value
        calls.append({"function": {"name": name, "arguments": args}})
    return calls


def tool_calls_of(message):
    calls = message.get("tool_calls") or parse_xml_tool_calls(message.get("content"))
    result = []
    for c in calls:
        fn = c["function"]
        args = fn["arguments"]
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = None
        result.append({"name": fn["name"], "arguments": args})
    return result


def grade_tool(task, message):
    calls = tool_calls_of(message)
    if not calls:
        return False, "no tool call"
    first, expected = calls[0], task["expected_call"]
    if first["name"] != expected["name"]:
        return False, first["name"]
    if first["arguments"] is None:
        return False, "invalid JSON arguments"
    ok = all(first["arguments"].get(k) == v for k, v in expected.get("arguments", {}).items())
    return ok, json.dumps(first["arguments"], ensure_ascii=False)


def grade(task, message, exec_timeout=30):
    """message: {"content": ..., "tool_calls": ...}. Returns (ok, detail, content without thinking)."""
    content = strip_thinking(message.get("content"))
    kind = task["type"]
    if kind == "math":
        ok, detail = grade_math(task, content)
    elif kind == "numeric":
        ok, detail = grade_numeric(task, content)
    elif kind == "mcq":
        ok, detail = grade_mcq(task, content)
    elif kind == "python":
        ok, detail = grade_python(task, content, exec_timeout)
    elif kind == "pytest":
        ok, detail = grade_pytest(task, content, exec_timeout)
    elif kind == "tool":
        ok, detail = grade_tool(task, {**message, "content": content})
    else:
        raise ValueError(f"Unknown task type: {kind}")
    return ok, detail, content


# ---------- main ----------

def load_tasks(paths, limit):
    tasks = []
    for p in paths:
        file_tasks = [json.loads(line) for line in Path(p).read_text(encoding="utf-8").splitlines() if line.strip()]
        for t in file_tasks[:limit] if limit else file_tasks:
            t["_source"] = Path(p).stem
            tasks.append(t)
    return tasks


def write_summary(results_dir, stamp, label, model, stats):
    """stats: {source: {"n", "correct", "tokens", "seconds"}} -> appends to summary.csv and prints."""
    summary_path = Path(results_dir) / "summary.csv"
    new_file = not summary_path.exists()
    with summary_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["time", "label", "model", "source", "n", "correct", "accuracy", "avg_tokens", "seconds"])
        print("\nSummary:")
        for key, s in stats.items():
            acc = s["correct"] / s["n"]
            writer.writerow([stamp, label, model, key, s["n"], s["correct"], f"{acc:.3f}",
                             s["tokens"] // s["n"], round(s["seconds"])])
            print(f"  {key:<20} {s['correct']}/{s['n']} = {acc:.1%}  (avg {s['tokens'] // s['n']} tok)")


def main():
    if hasattr(sys.stdout, "reconfigure"):  # for non-ASCII output on the Windows console (not on Colab)
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", required=True, help="JSONL task files")
    ap.add_argument("--label", required=True, help="Name of this run, e.g. base / sft-v1")
    ap.add_argument("--base-url", default="http://localhost:1234/v1")
    ap.add_argument("--model", help="Default: the first model on the server")
    ap.add_argument("--system", default="")
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="Max tasks per file (0 = all)")
    ap.add_argument("--timeout", type=int, default=3600, help="Seconds per request")
    ap.add_argument("--exec-timeout", type=int, default=30, help="Seconds for Python tests")
    args = ap.parse_args()

    if not args.model:
        models = requests.get(f"{args.base_url}/models", timeout=10).json()["data"]
        args.model = models[0]["id"]
    print(f"Model: {args.model}")

    tasks = load_tasks(args.tasks, args.limit)
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = RESULTS_DIR / f"{args.label}_{stamp}.jsonl"

    stats = {}
    with out_path.open("w", encoding="utf-8") as out:
        for i, task in enumerate(tasks, 1):
            start = time.time()
            message, tokens = chat(args, build_messages(task, args.system), tools=task.get("tools"))
            ok, detail, content = grade(task, message, args.exec_timeout)
            elapsed = time.time() - start
            key = task["_source"]
            s = stats.setdefault(key, {"n": 0, "correct": 0, "tokens": 0, "seconds": 0.0})
            s["n"] += 1
            s["correct"] += int(ok)
            s["tokens"] += tokens
            s["seconds"] += elapsed
            out.write(json.dumps({"id": task["id"], "source": key, "type": task["type"], "ok": ok,
                                  "detail": detail, "tokens": tokens, "seconds": round(elapsed, 1),
                                  "response": content}, ensure_ascii=False) + "\n")
            out.flush()
            print(f"[{i}/{len(tasks)}] {key}/{task['id']}: {'OK ' if ok else 'ERR'} "
                  f"({tokens} tok, {elapsed:.0f}s)")

    write_summary(RESULTS_DIR, stamp, args.label, args.model, stats)
    print(f"\nDetails: {out_path}")


if __name__ == "__main__":
    main()
