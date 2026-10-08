"""Multi-turn tool-use (agent) episodes: reasoning with a Python tool and coding in a sandbox.

An Episode holds the conversation, the tools and the sandbox directory for one task. Model output is fed in with
consume(): tool calls are actually executed in the sandbox and the results are appended as "tool" messages.
The episode ends when the model answers without calling a tool, and is graded with grade_episode().

Task types:
  math / numeric / mcq : run_eval.py tasks; run_python is offered as a tool, the answer is still \\boxed{} / "Answer: X"
  agent_code           : {"id", "type": "agent_code", "prompt", "tests", "stub"}; read_file / write_file / run_tests

Warning: the tools are not a real sandbox; model code runs on this machine (on Colab, on the VM).
"""

import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import run_eval as R

OUTPUT_LIMIT = 3000
PY_TIMEOUT = 20
TEST_TIMEOUT = 60

PYTHON_TOOL = {"type": "function", "function": {
    "name": "run_python",
    "description": "Run a Python 3 script in a fresh process and return its stdout and stderr. "
                   "State is not kept between calls, so print everything you need. numpy, sympy and scipy are available.",
    "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "Python source code"}},
                   "required": ["code"]}}}

CODE_TOOLS = [
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a text file from the project directory.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file", "description": "Create or overwrite a text file in the project directory.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                       "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "run_tests", "description": "Run the project's test suite with pytest and return the output.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
]

PYTHON_HINT = "\n\nYou can call the run_python tool to compute or verify intermediate results."

CODE_PROMPT = (
    "You are working in a Python project directory. `test_solution.py` contains the tests and `solution.py` "
    "is a stub.\n\nTask:\n{prompt}\n\nUse the tools to implement `solution.py`, run the tests and fix any "
    "failures. Do not modify the tests. When all tests pass, reply with a short summary of your solution.")


def truncate(text, limit=OUTPUT_LIMIT):
    if len(text) <= limit:
        return text
    head = limit // 3
    return text[:head] + "\n...[truncated]...\n" + text[-(limit - head):]


class Sandbox:
    def __init__(self, files):
        self.dir = Path(tempfile.mkdtemp(prefix="agent_")).resolve()
        for name, text in files.items():
            (self.dir / name).write_text(text, encoding="utf-8")

    def _path(self, path):
        p = (self.dir / path).resolve()
        if p != self.dir and self.dir not in p.parents:
            raise ValueError(f"path outside the project directory: {path}")
        return p

    def run(self, cmd, timeout):
        """(exit code, truncated output)"""
        try:
            proc = subprocess.run(cmd, cwd=self.dir, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return None, f"Error: timed out after {timeout}s"
        return proc.returncode, truncate(proc.stdout + proc.stderr)

    def call(self, name, args):
        # Tool errors are returned to the model as text (the agent should see and fix them)
        try:
            if name == "run_python":
                code, out = self.run([sys.executable, "-c", args["code"]], PY_TIMEOUT)
                return out if code is None else f"{out}\n[exit code {code}]".lstrip()
            if name == "read_file":
                return truncate(self._path(args["path"]).read_text(encoding="utf-8"))
            if name == "write_file":
                p = self._path(args["path"])
                if p.name.startswith("test_"):
                    return "Error: test files are read-only."
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(args["content"], encoding="utf-8")
                return f"Wrote {len(args['content'])} characters to {args['path']}."
            if name == "run_tests":
                code, out = self.run(self.pytest_cmd(), TEST_TIMEOUT)
                return out if code is None else f"{out}\n[exit code {code}]".lstrip()
            return f"Error: unknown tool '{name}'"
        except (KeyError, TypeError) as e:
            return f"Error: invalid arguments for {name}: {e}"
        except (OSError, ValueError, UnicodeDecodeError) as e:
            return f"Error: {type(e).__name__}: {e}"

    @staticmethod
    def pytest_cmd():
        return [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider"]

    def close(self):
        shutil.rmtree(self.dir, ignore_errors=True)


def parse_tool_calls(text, tools):
    """Qwen3.5/3.6 XML calls; parameters typed as string in the schema stay raw text (so code content is not mangled)."""
    schemas = {t["function"]["name"]: t["function"]["parameters"].get("properties", {}) for t in tools}
    calls = []
    for name, body in re.findall(r"<function=([^>\s]+)>(.*?)</function>", text or "", flags=re.S):
        props = schemas.get(name, {})
        args = {}
        for key, value in re.findall(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", body, flags=re.S):
            if props.get(key, {}).get("type") == "string":
                args[key] = value
            else:
                try:
                    args[key] = json.loads(value)
                except json.JSONDecodeError:
                    args[key] = value
        calls.append({"name": name, "arguments": args})
    return calls


class Episode:
    def __init__(self, task, rollout=0, max_turns=8):
        self.task, self.rollout, self.max_turns = task, rollout, max_turns
        if task["type"] == "agent_code":
            self.tools = CODE_TOOLS
            self.sandbox = Sandbox({"test_solution.py": task["tests"], "solution.py": task["stub"]})
            user = CODE_PROMPT.format(prompt=task["prompt"])
        else:
            self.tools = [PYTHON_TOOL]
            self.sandbox = Sandbox({})
            user = R.build_messages(task)[-1]["content"] + PYTHON_HINT
        self.messages = [{"role": "user", "content": user}]
        self.done, self.failed = False, None
        self.turns = self.tokens = self.n_tool_calls = 0

    def consume(self, text, finish_reason="stop", n_tokens=0):
        """One assistant turn (raw vLLM output: '...thinking...</think>answer/tool calls')."""
        self.turns += 1
        self.tokens += n_tokens
        if finish_reason != "stop":
            self.done, self.failed = True, "length"
            return
        reasoning, sep, answer = text.partition("</think>")
        if not sep:
            reasoning, answer = "", text
        calls = parse_tool_calls(answer, self.tools)
        msg = {"role": "assistant", "content": answer.split("<tool_call>")[0].strip(),
               "reasoning_content": reasoning.strip()}
        if calls:
            msg["tool_calls"] = [{"type": "function", "function": c} for c in calls]
        self.messages.append(msg)
        if not calls:
            self.done = True
            return
        for c in calls:
            self.n_tool_calls += 1
            self.messages.append({"role": "tool", "content": self.sandbox.call(c["name"], c["arguments"])})
        if self.turns >= self.max_turns:
            self.done, self.failed = True, "turns"

    def fail(self, reason):
        self.done, self.failed = True, reason


def grade_episode(ep):
    """(ok, detail). For code tasks the tests are restored to their original content and run again."""
    if ep.failed:
        return False, ep.failed
    if ep.task["type"] == "agent_code":
        (ep.sandbox.dir / "test_solution.py").write_text(ep.task["tests"], encoding="utf-8")
        code, out = ep.sandbox.run(Sandbox.pytest_cmd(), TEST_TIMEOUT)
        return code == 0, out[-300:]
    ok, detail, _ = R.grade(ep.task, {"content": ep.messages[-1]["content"]})
    return ok, detail


def make_stub(test_info):
    """KodCode test_info -> 'def f(x):\\n    raise NotImplementedError' (functions only)."""
    defs = [ti["function_declaration"].strip() for ti in test_info
            if ti.get("function_declaration", "").strip().startswith("def ")]
    return "\n\n\n".join(f"{d.rstrip(':')}:\n    raise NotImplementedError\n" for d in defs)
