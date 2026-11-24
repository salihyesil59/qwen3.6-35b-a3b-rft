"""Coding tasks solved inside Claude Code, driven by a local model on llama-server.

Claude Code speaks the Anthropic Messages API. The Proxy in this module sits between Claude Code and llama-server:
it converts every request to an OpenAI chat completion, sends it to llama-server (the model's own chat template, tool
calls as XML, the thinking apart from the answer), converts the reply back, and keeps the last exchange of every run.
That exchange holds the whole conversation in OpenAI format, ready for the chat template: it is the training sample.

Claude Code runs headless (`claude -p`) in a fresh project folder per run, as an unprivileged user, with the same
six tools that are used at home (CLAUDE_TOOLS). A run is graded by the task's tests after the protected files
(the tests, and for JavaScript the package config) are restored, so editing the tests does not help.

Task: {"id", "source", "prompt", "files": {path: text}, "protected": [paths], "grade": shell command,
       "node_modules": optional shared folder that is linked into the project}

Warning: the model's commands run on this machine (on Colab, on the VM), only as a separate user.

At home, the same proxy puts Claude Code in front of LM Studio or llama-server, so the model sees its requests in
exactly the format it was trained on:
    python claude_code.py --upstream http://localhost:1234/v1 --model <model id in LM Studio>
    ANTHROPIC_BASE_URL=http://127.0.0.1:8100/r/home ANTHROPIC_API_KEY=local claude --tools Bash,Read,Edit,Write,Glob,Grep
"""

import argparse
import collections
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import requests

CLAUDE_TOOLS = "Bash,Read,Edit,Write,Glob,Grep"
MODEL_NAME = "qwen3.6-35b-a3b"        # the name Claude Code sends; llama-server serves one model whatever the name
GRADE_TIMEOUT = 180
CONFIG_FILES = {"conftest.py", "pytest.ini", "pytest.toml", ".pytest.toml", "tox.ini", "setup.cfg", "pyproject.toml", "jest.config.js",
                "jest.config.cjs", "jest.config.mjs", "jest.config.json", ".babelrc", "babel.config.cjs"}

KODCODE_PROMPT = ("{question}\n\nThe project folder has solution.py with the signatures of the functions to write, and "
                  "test_solution.py with tests for them. Implement the functions in solution.py; do not change the tests.")

POLYGLOT_PROMPT = ("{instructions}\n\n####\n\nUse the above instructions to modify the supplied files: {files}\n"
                   "Don't change the names of existing functions or classes, as they may be referenced from other code "
                   "like unit tests, etc.\nOnly use standard libraries, don't suggest installing any packages.\n"
                   "The tests are in {tests}; do not change them. Run them to check your work.")


# ---------------------------------------------------------------- Anthropic <-> OpenAI

def _text_of(content):
    if isinstance(content, str):
        return content
    parts = []
    for b in content or []:
        if b.get("type") == "text":
            parts.append(b["text"])
        elif b.get("type") == "image":
            parts.append("[image]")
    return "\n\n".join(parts)


def _key(text):
    return hashlib.md5(text.strip().encode()).hexdigest()


def to_openai(body, reasoning_cache=None):
    """Anthropic request body -> {"messages", "tools"} in OpenAI format (tool call arguments as JSON strings)."""
    cache = reasoning_cache or {}
    system = body.get("system") or ""
    if isinstance(system, list):
        system = "\n\n".join(b["text"] for b in system
                             if b.get("type") == "text" and not b["text"].startswith("x-anthropic-billing-header"))
    systems, messages = [system] if system else [], []
    for m in body.get("messages", []):
        content = m["content"] if isinstance(m["content"], list) else [{"type": "text", "text": m["content"]}]
        if m["role"] == "system":
            systems.append(_text_of(content))
        elif m["role"] == "user":
            for b in content:
                if b.get("type") == "tool_result":
                    messages.append({"role": "tool", "tool_call_id": b.get("tool_use_id", ""),
                                     "content": _text_of(b.get("content"))})
            text = _text_of([b for b in content if b.get("type") in ("text", "image")])
            if text:
                messages.append({"role": "user", "content": text})
        elif m["role"] == "assistant":
            thinking = "\n\n".join(b.get("thinking", "") for b in content if b.get("type") == "thinking").strip()
            text = "\n\n".join(b["text"] for b in content if b.get("type") == "text").strip()
            calls = [{"id": b["id"], "type": "function",
                      "function": {"name": b["name"], "arguments": json.dumps(b.get("input", {}), ensure_ascii=False)}}
                     for b in content if b.get("type") == "tool_use"]
            if not thinking:   # history that comes back without its thinking: the proxy kept it
                thinking = cache.get(calls[0]["id"] if calls else _key(text), "")
            msg = {"role": "assistant", "content": text}
            if thinking:
                msg["reasoning_content"] = thinking
            if calls:
                msg["tool_calls"] = calls
            messages.append(msg)
    if systems:
        messages.insert(0, {"role": "system", "content": "\n\n".join(systems)})
    tools = [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
                                               "parameters": t.get("input_schema") or {"type": "object", "properties": {}}}}
             for t in body.get("tools", []) if t.get("name") and t.get("type") in (None, "custom")]
    return {"messages": messages, "tools": tools}


def to_anthropic_content(msg):
    """OpenAI assistant message -> Anthropic content blocks."""
    blocks = []
    if msg.get("reasoning_content"):
        blocks.append({"type": "thinking", "thinking": msg["reasoning_content"], "signature": "local"})
    if (msg.get("content") or "").strip():
        blocks.append({"type": "text", "text": msg["content"]})
    for c in msg.get("tool_calls") or []:
        try:
            args = json.loads(c["function"]["arguments"] or "{}")
        except json.JSONDecodeError:
            args = {"_unparsed_arguments": c["function"]["arguments"]}
        blocks.append({"type": "tool_use", "id": c.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                       "name": c["function"]["name"], "input": args if isinstance(args, dict) else {"value": args}})
    return blocks


def for_template(messages):
    """Conversation for the chat template: tool call arguments as objects, no empty thinking."""
    out = []
    for m in messages:
        m = {k: v for k, v in m.items() if not (k == "reasoning_content" and not v) and k != "tool_call_id"}
        if m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                args = c["function"]["arguments"]
                calls.append({"type": "function", "function": {
                    "name": c["function"]["name"], "arguments": json.loads(args or "{}") if isinstance(args, str) else args}})
            m["tool_calls"] = calls
        out.append(m)
    return out


def is_clean(messages):
    """No user message after the first reply. The chat template drops the thinking of every reply before the last
    user message, so a reply before such a message would be trained without its thinking."""
    first = next((i for i, m in enumerate(messages) if m["role"] == "assistant"), len(messages))
    return all(m["role"] != "user" for m in messages[first:])


# ---------------------------------------------------------------- proxy

class Proxy:
    """Anthropic Messages API on 127.0.0.1:port -> llama-server's OpenAI API. Base URL of a run: base_url(run_id)."""

    def __init__(self, upstream="http://127.0.0.1:8000/v1", port=8100, sampling=None, turn_tokens=16384, seed=3407,
                 model=MODEL_NAME):
        self.upstream, self.port, self.turn_tokens, self.seed = upstream.rstrip("/"), port, turn_tokens, seed
        self.model = model
        self.sampling = sampling or {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
        self.runs, self.reasoning, self.lock = {}, {}, threading.Lock()
        self.http = requests.Session()
        self.http.mount("http://", requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=128))
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                m = re.match(r"^/r/([^/]+)(/v1/messages(/count_tokens)?)$", self.path.split("?")[0])
                if not m:
                    return self.send_json(404, {"type": "error", "error": {"type": "not_found_error", "message": self.path}})
                if m.group(3):
                    return self.send_json(200, {"input_tokens": len(json.dumps(body)) // 4})
                status, payload = proxy.complete(m.group(1), body)
                if status != 200 or not body.get("stream"):
                    return self.send_json(status, payload)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                for event, data in sse_events(payload):
                    self.wfile.write(f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode())
                self.wfile.flush()
                self.close_connection = True

            def do_GET(self):
                self.send_json(404, {"type": "error", "error": {"type": "not_found_error", "message": self.path}})

            def send_json(self, status, payload):
                data = json.dumps(payload, ensure_ascii=False).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def base_url(self, run_id):
        return f"http://127.0.0.1:{self.port}/r/{run_id}"

    def complete(self, run_id, body):
        with self.lock:
            req = to_openai(body, self.reasoning)
            run = self.runs.setdefault(run_id, {"requests": 0, "tokens": 0, "last": None, "errors": []})
            run["requests"] += 1
        payload = {"model": self.model, "messages": req["messages"],
                   "max_tokens": min(int(body.get("max_tokens") or self.turn_tokens), self.turn_tokens),
                   # every run (task + sample) and every turn gets its own seed: two samples of a task differ
                   "seed": (self.seed + int(_key(run_id)[:7], 16) + 7919 * run["requests"]) % 2**31, **self.sampling}
        if req["tools"]:
            payload["tools"] = req["tools"]
        try:
            r = self.http.post(f"{self.upstream}/chat/completions", json=payload, timeout=3600)
        except requests.RequestException as e:
            return self.error(run, 500, "api_error", f"llama-server: {e}")
        if r.status_code != 200:
            # 4xx (e.g. the conversation no longer fits the slot's context) is not retried by Claude Code
            return self.error(run, 400 if r.status_code < 500 else 500,
                              "invalid_request_error" if r.status_code < 500 else "api_error", r.text[:500])
        data = r.json()
        choice, usage = data["choices"][0], data.get("usage") or {}
        msg = {"role": "assistant", "content": choice["message"].get("content") or ""}
        reasoning = choice["message"].get("reasoning_content") or choice["message"].get("reasoning")   # LM Studio: "reasoning"
        if reasoning:
            msg["reasoning_content"] = reasoning
        if choice["message"].get("tool_calls"):
            msg["tool_calls"] = [{"id": c.get("id") or f"toolu_{uuid.uuid4().hex[:24]}", "type": "function",
                                  "function": {"name": c["function"]["name"], "arguments": c["function"]["arguments"]}}
                                 for c in choice["message"]["tool_calls"]]
        finish = choice.get("finish_reason")
        with self.lock:
            run["tokens"] += usage.get("completion_tokens", 0)
            if req["tools"]:   # the agent loop (side requests without tools are not the conversation)
                run["last"] = {"messages": req["messages"] + [msg], "tools": req["tools"], "finish": finish}
            if msg.get("reasoning_content"):
                key = msg["tool_calls"][0]["id"] if msg.get("tool_calls") else _key(msg["content"])
                self.reasoning[key] = msg["reasoning_content"]
        content = to_anthropic_content(msg)
        stop = "tool_use" if msg.get("tool_calls") else "max_tokens" if finish == "length" else "end_turn"
        return 200, {"id": f"msg_{uuid.uuid4().hex[:24]}", "type": "message", "role": "assistant",
                     "model": body.get("model", MODEL_NAME), "content": content, "stop_reason": stop,
                     "stop_sequence": None, "usage": {"input_tokens": usage.get("prompt_tokens", 0),
                                                      "output_tokens": usage.get("completion_tokens", 0)}}

    def error(self, run, status, kind, message):
        with self.lock:
            run["errors"].append(message[:300])
        return status, {"type": "error", "error": {"type": kind, "message": message}}

    def pop(self, run_id):
        with self.lock:
            return self.runs.pop(run_id, {"requests": 0, "tokens": 0, "last": None, "errors": []})

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def sse_events(message):
    """A complete Anthropic message as the event stream of a streamed one."""
    start = {**message, "content": [], "stop_reason": None,
             "usage": {"input_tokens": message["usage"]["input_tokens"], "output_tokens": 1}}
    yield "message_start", {"type": "message_start", "message": start}
    for i, b in enumerate(message["content"]):
        if b["type"] == "thinking":
            yield "content_block_start", {"type": "content_block_start", "index": i,
                                          "content_block": {"type": "thinking", "thinking": "", "signature": ""}}
            yield "content_block_delta", {"type": "content_block_delta", "index": i,
                                          "delta": {"type": "thinking_delta", "thinking": b["thinking"]}}
            yield "content_block_delta", {"type": "content_block_delta", "index": i,
                                          "delta": {"type": "signature_delta", "signature": b["signature"]}}
        elif b["type"] == "text":
            yield "content_block_start", {"type": "content_block_start", "index": i,
                                          "content_block": {"type": "text", "text": ""}}
            yield "content_block_delta", {"type": "content_block_delta", "index": i,
                                          "delta": {"type": "text_delta", "text": b["text"]}}
        else:
            yield "content_block_start", {"type": "content_block_start", "index": i,
                                          "content_block": {"type": "tool_use", "id": b["id"], "name": b["name"], "input": {}}}
            yield "content_block_delta", {"type": "content_block_delta", "index": i,
                                          "delta": {"type": "input_json_delta",
                                                    "partial_json": json.dumps(b["input"], ensure_ascii=False)}}
        yield "content_block_stop", {"type": "content_block_stop", "index": i}
    yield "message_delta", {"type": "message_delta", "delta": {"stop_reason": message["stop_reason"], "stop_sequence": None},
                            "usage": {"output_tokens": message["usage"]["output_tokens"]}}
    yield "message_stop", {"type": "message_stop"}


# ---------------------------------------------------------------- llama-server

def download_gguf(repo, pattern, local_dir="/content/gguf"):
    """The model file of a GGUF repo whose name contains `pattern` (all parts of a split file); returns the first part."""
    from huggingface_hub import hf_hub_download, list_repo_files
    files = sorted(f for f in list_repo_files(repo)
                   if f.endswith(".gguf") and pattern in f and "mmproj" not in f
                   and not f.endswith("imatrix.gguf"))   # the imatrix itself; "...-imatrix-Q4_K_M.gguf" is a model
    assert files, f"no .gguf with '{pattern}' in {repo}: {[f for f in list_repo_files(repo) if f.endswith('.gguf')]}"
    first = files[0]
    split = re.match(r"(.*)-00001-of-\d+\.gguf$", first)
    parts = [f for f in files if split and f.startswith(split.group(1))] if split else [first]
    paths = [hf_hub_download(repo, f, local_dir=local_dir) for f in parts]
    return paths[0]


def start_llama_server(binary, model_path, slots, slot_context, port=8000, seed=3407, log="/content/llama-server.log"):
    proc = subprocess.Popen(
        [binary, "-m", model_path, "-a", MODEL_NAME, "--host", "127.0.0.1", "--port", str(port),
         "-ngl", "999",                                   # the whole model on the GPU
         "-np", str(slots), "-c", str(slots * slot_context),   # the context is shared out among the slots
         "-fa", "on",
         "--jinja",                                       # the model's own chat template: tool calls as XML
         "--reasoning-format", "deepseek",                # the thinking comes back apart from the answer
         "--seed", str(seed)],
        stdout=open(log, "w"), stderr=subprocess.STDOUT)
    for _ in range(180):
        if proc.poll() is not None:
            raise RuntimeError("llama-server stopped:\n" + open(log).read()[-3000:])
        try:
            if requests.get(f"http://127.0.0.1:{port}/health", timeout=5).ok:
                return proc
        except requests.RequestException:
            pass
        time.sleep(10)
    proc.terminate()
    raise RuntimeError("llama-server did not start in 30 min:\n" + open(log).read()[-3000:])


# ---------------------------------------------------------------- tasks

def kodcode_task(prefix, row, stub, source):
    """A KodCode row as a task; the id is the notebooks' uid(prefix, question_id)."""
    return {"id": f"{prefix}-{hashlib.md5(row['question_id'].encode()).hexdigest()[:12]}", "source": source,
            "prompt": KODCODE_PROMPT.format(question=row["question"]),
            "files": {"solution.py": stub, "test_solution.py": row["test"]},
            "protected": ["test_solution.py"],
            "grade": "python3 -m pytest -q -p no:cacheprovider test_solution.py"}


def polyglot_tasks(repo_dir, lang, node_modules=None):
    """Exercism exercises of the Aider polyglot benchmark; the reference solution (.meta) is left out."""
    tasks = []
    for ex in sorted((Path(repo_dir) / lang / "exercises" / "practice").iterdir()):
        cfg = json.loads((ex / ".meta" / "config.json").read_text(encoding="utf-8"))
        docs = [ex / ".docs" / n for n in ("introduction.md", "instructions.md", "instructions.append.md")]
        instructions = "\n\n".join(p.read_text(encoding="utf-8").strip() for p in docs if p.exists())
        files = {}
        for p in sorted(ex.rglob("*")):
            rel = p.relative_to(ex).as_posix()
            if p.is_file() and not rel.startswith((".meta/", ".docs/", "node_modules/")):
                files[rel] = p.read_text(encoding="utf-8")
        tests, solution = cfg["files"]["test"], cfg["files"]["solution"]
        if lang == "javascript":
            for t in tests:   # Exercism skips all but the first test; the benchmark runs them all
                files[t] = re.sub(r"\bxtest\(", "test(", re.sub(r"\bxit\(", "it(", files[t]))
            protected = tests + [f for f in ("package.json", "babel.config.js", ".npmrc") if f in files]
            grade = f"./node_modules/.bin/jest --ci {' '.join(tests)}"
        else:
            protected = list(tests)
            grade = f"python3 -m pytest -q -p no:cacheprovider {' '.join(tests)}"
        task = {"id": f"polyglot-{lang}-{ex.name}", "source": f"polyglot_{lang}",
                "prompt": POLYGLOT_PROMPT.format(instructions=instructions, files=" ".join(solution), tests=" ".join(tests)),
                "files": files, "protected": protected, "grade": grade}
        if node_modules:
            task["node_modules"] = str(node_modules)
        tasks.append(task)
    return tasks


def js_dev_dependencies(repo_dir):
    """package.json for one shared node_modules with the devDependencies of the exercises (they use the same versions)."""
    deps = {}
    for p in sorted((Path(repo_dir) / "javascript" / "exercises" / "practice").glob("*/package.json")):
        deps.update(json.loads(p.read_text(encoding="utf-8")).get("devDependencies", {}))
    return {"name": "polyglot-js-deps", "private": True, "devDependencies": deps}


# ---------------------------------------------------------------- runs

def ensure_user(name="agent"):
    """An unprivileged user for Claude Code and the model's commands; returns its uid."""
    import pwd
    try:
        return pwd.getpwnam(name).pw_uid
    except KeyError:
        subprocess.run(["useradd", "-m", "-s", "/bin/bash", name], check=True)
        return pwd.getpwnam(name).pw_uid


def _chown(path, user):
    if user:
        subprocess.run(["chown", "-R", f"{user}:{user}", str(path)], check=True)


def _run(cmd, cwd, env, timeout, user, shell=False):
    """(exit code or None on timeout, output). The whole process group is killed on timeout."""
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, user=user, shell=shell, start_new_session=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        out, _ = proc.communicate(timeout=timeout)
        return proc.returncode, out
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        out, _ = proc.communicate()
        return None, out


def grade(task, project, env, user=None):
    """(ok, detail): the protected files are restored and config files the model added are removed first."""
    for rel in task["protected"]:
        (project / rel).write_text(task["files"][rel], encoding="utf-8")
    for dirpath, dirnames, filenames in os.walk(project):   # does not follow the node_modules link
        dirnames[:] = [d for d in dirnames if d != "node_modules"]
        for name in filenames:
            p = Path(dirpath) / name
            if name in CONFIG_FILES and p.relative_to(project).as_posix() not in task["files"]:
                p.unlink(missing_ok=True)
    shutil.rmtree(project / "__pycache__", ignore_errors=True)
    _chown(project, user)
    code, out = _run(task["grade"], project, env, GRADE_TIMEOUT, user, shell=True)
    return code == 0, (out or "")[-300:] if code is not None else "grade timed out"


def run_one(task, sample, proxy, work_root, claude_bin, max_turns, minutes, user=None, keep_conversation=False):
    run_id = f"{task['id']}-{sample}"
    root = Path(work_root) / run_id
    shutil.rmtree(root, ignore_errors=True)
    project, home = root / "project", root / "home"
    for rel, text in task["files"].items():
        (project / rel).parent.mkdir(parents=True, exist_ok=True)
        (project / rel).write_text(text, encoding="utf-8")
    home.mkdir(parents=True)
    if task.get("node_modules"):
        (project / "node_modules").symlink_to(task["node_modules"])
    _chown(root, user)
    env = {"PATH": f"{Path(claude_bin).parent}:{os.environ.get('PATH', '/usr/bin:/bin')}", "HOME": str(home),
           "LANG": "C.UTF-8", "ANTHROPIC_BASE_URL": proxy.base_url(run_id), "ANTHROPIC_API_KEY": "local",
           "ANTHROPIC_DEFAULT_HAIKU_MODEL": MODEL_NAME, "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
           "DISABLE_AUTOUPDATER": "1", "IS_SANDBOX": "1"}
    cmd = [claude_bin, "-p", task["prompt"], "--model", MODEL_NAME, "--tools", CLAUDE_TOOLS,
           "--max-turns", str(max_turns), "--output-format", "json", "--dangerously-skip-permissions"]
    t0 = time.time()
    code, out = _run(cmd, project, env, minutes * 60, user)
    seconds = time.time() - t0
    log = proxy.pop(run_id)
    result = None
    for line in reversed((out or "").strip().splitlines()):
        try:
            result = json.loads(line)
            break
        except json.JSONDecodeError:
            continue
    if code is None:
        end = "timeout"
    elif isinstance(result, dict) and result.get("subtype"):
        end = result["subtype"]                       # success / error_max_turns / error_during_execution
    else:
        end = f"exit {code}: {(out or '').strip()[-200:]}"
    if log["last"] and log["last"]["finish"] == "length":
        end = "length"
    if log["errors"]:   # e.g. the conversation no longer fit the slot's context
        end = f"server error: {log['errors'][-1][:150]}"
    ok, detail = grade(task, project, {**env, "ANTHROPIC_API_KEY": ""}, user)
    messages = log["last"]["messages"] if log["last"] else []
    record = {"id": task["id"], "source": task["source"], "sample": sample, "ok": ok, "end": end, "detail": detail,
              "turns": sum(m["role"] == "assistant" for m in messages), "requests": log["requests"],
              "tool_calls": sum(len(m.get("tool_calls") or []) for m in messages), "tokens": log["tokens"],
              "seconds": round(seconds, 1)}
    conversation = None
    if keep_conversation and log["last"]:
        conversation = {"id": task["id"], "sample": sample, "ok": ok, "end": end,
                        "messages": messages, "tools": log["last"]["tools"]}
    shutil.rmtree(root, ignore_errors=True)
    return record, conversation


def run_tasks(tasks, samples, proxy, results_path, claude_bin, work_root="/content/cc_work", concurrency=16,
              max_turns=40, minutes=20, hours=None, user=None, conversations_path=None, label=""):
    """Runs every (task, sample) not yet in results_path, `concurrency` at a time; appends results as they finish.
    Resumable: after a disconnect call it again. Stops starting new runs after `hours`."""
    done = set()
    if os.path.exists(results_path):
        for l in open(results_path, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                done.add((r["id"], r["sample"]))
    todo = [(t, k) for k in range(samples) for t in tasks if (t["id"], k) not in done]
    print(f"{label} done: {len(done)}, remaining: {len(todo)}")
    start, finished, solved = time.time(), 0, 0
    write_lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        pending, queue = set(), iter(todo)
        while True:
            while len(pending) < concurrency and not (hours and time.time() - start > hours * 3600):
                item = next(queue, None)
                if item is None:
                    break
                pending.add(ex.submit(run_one, item[0], item[1], proxy, work_root, claude_bin, max_turns, minutes,
                                      user, conversations_path is not None))
            if not pending:
                break
            ready, pending = wait(pending, return_when=FIRST_COMPLETED)
            for f in ready:
                record, conversation = f.result()
                with write_lock:
                    with open(results_path, "a", encoding="utf-8") as out:
                        out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    if conversation is not None:
                        with open(conversations_path, "a", encoding="utf-8") as out:
                            out.write(json.dumps(conversation, ensure_ascii=False) + "\n")
                finished += 1
                solved += record["ok"]
                if finished % 10 == 0 or finished == len(todo):
                    h = (time.time() - start) / 3600
                    print(f"{label} {finished}/{len(todo)} | solved {solved / finished:.0%} | {h:.2f} h")
    if hours and time.time() - start > hours * 3600:
        print("Time budget reached: runs not started yet are left for the next call.")


def summarize(results_path):
    """Per source: runs, solved, average turns / tool calls / tokens / seconds, end reasons."""
    st = collections.defaultdict(lambda: {"n": 0, "correct": 0, "tokens": 0, "seconds": 0.0, "turns": 0,
                                          "tool_calls": 0, "ends": collections.Counter()})
    for l in open(results_path, encoding="utf-8"):
        if not l.strip():
            continue
        r = json.loads(l)
        s = st[r["source"]]
        s["n"] += 1
        s["correct"] += int(r["ok"])
        s["tokens"] += r["tokens"]
        s["seconds"] += r["seconds"]
        s["turns"] += r["turns"]
        s["tool_calls"] += r["tool_calls"]
        s["ends"][r["end"].split(":")[0].split(";")[0]] += 1
    return dict(st)


def main():
    ap = argparse.ArgumentParser(description="Claude Code (Anthropic API) -> an OpenAI-compatible server, in the format "
                                             "the model was trained on. Point Claude Code at http://127.0.0.1:PORT/r/home")
    ap.add_argument("--upstream", default="http://localhost:1234/v1", help="OpenAI API of LM Studio / llama-server")
    ap.add_argument("--model", default=MODEL_NAME, help="model id on the upstream server")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--max-tokens", type=int, default=16384, help="max tokens of one reply")
    args = ap.parse_args()
    proxy = Proxy(args.upstream, args.port, {"temperature": args.temperature, "top_p": 0.95, "top_k": 20},
                  args.max_tokens, model=args.model)
    print(f"ANTHROPIC_BASE_URL={proxy.base_url('home')} ANTHROPIC_API_KEY=local claude --tools {CLAUDE_TOOLS}")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        proxy.close()


if __name__ == "__main__":
    main()
