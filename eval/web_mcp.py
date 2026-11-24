"""Web search and page fetch for Claude Code on a local model, as a minimal MCP server (stdio, JSON-RPC 2.0).

Claude Code's own WebSearch runs on Anthropic's servers, so it does not work with a local model. This server gives
the model two ordinary tools, mcp__web__search and mcp__web__fetch, with the same definitions on Colab (training
data) and at home. No MCP SDK: the tool definitions below are exactly what the model sees.

    pip install requests ddgs==9.16.1
    claude --mcp-config <file with {"mcpServers": {"web": {"command": "python", "args": ["web_mcp.py"]}}}>
"""

import json
import re
import sys
from html.parser import HTMLParser

import requests

SEARCH_RESULTS = 8
PAGE_CHARS = 20000
TIMEOUT = 20

TOOLS = [
    {"name": "search",
     "description": "Search the web (DuckDuckGo). Returns the title, URL and a short snippet of each result. "
                    "Use fetch to read a result.",
     "inputSchema": {"type": "object", "properties": {"query": {"type": "string", "description": "Search query"}},
                     "required": ["query"]}},
    {"name": "fetch",
     "description": "Fetch a web page and return its text (HTML tags removed), at most 20000 characters from "
                    "`offset` on.",
     "inputSchema": {"type": "object", "properties": {
         "url": {"type": "string", "description": "http(s) URL"},
         "offset": {"type": "integer", "description": "Character offset to start from (default 0)"}},
         "required": ["url"]}},
]


class _Text(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "head"}
    BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "pre", "section", "article", "table"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.skip:
            self.skip -= 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)

    def text(self):
        text = re.sub(r"[ \t\r\f\v]+", " ", "".join(self.parts))
        return re.sub(r"\n\s*\n+", "\n\n", text).strip()


def search(query):
    from ddgs import DDGS
    results = DDGS().text(query, max_results=SEARCH_RESULTS)
    if not results:
        return "No results."
    return "\n\n".join(f"{i}. {r.get('title', '')}\n{r.get('href', '')}\n{r.get('body', '')}"
                       for i, r in enumerate(results, 1))


def fetch(url, offset=0):
    if not re.match(r"^https?://", url):
        raise ValueError("only http(s) URLs")
    r = requests.get(url, timeout=TIMEOUT, headers={"User-Agent": "Mozilla/5.0 (compatible; web-mcp)"})
    r.raise_for_status()
    if "html" in r.headers.get("Content-Type", "html"):
        parser = _Text()
        parser.feed(r.text)
        text = parser.text()
    else:
        text = r.text
    offset = max(int(offset or 0), 0)
    part = text[offset:offset + PAGE_CHARS]
    rest = len(text) - offset - len(part)
    return part + (f"\n\n[{rest} more characters: fetch again with offset={offset + len(part)}]" if rest > 0 else "")


def call(name, args):
    if name == "search":
        return search(args["query"])
    if name == "fetch":
        return fetch(args["url"], args.get("offset", 0))
    raise ValueError(f"unknown tool: {name}")


def handle(msg):
    method, params = msg.get("method"), msg.get("params") or {}
    if method == "initialize":
        return {"protocolVersion": params.get("protocolVersion", "2025-06-18"), "capabilities": {"tools": {}},
                "serverInfo": {"name": "web", "version": "1.0"}}
    if method == "tools/list":
        return {"tools": TOOLS}
    if method == "tools/call":
        try:
            text, error = call(params.get("name"), params.get("arguments") or {}), False
        except Exception as e:   # returned to the model as a tool error, so it can try something else
            text, error = f"Error: {type(e).__name__}: {e}", True
        return {"content": [{"type": "text", "text": text}], "isError": error}
    if method == "ping":
        return {}
    raise KeyError(method)


def main():
    sys.stdin.reconfigure(encoding="utf-8")    # Windows would read and write the console code page
    sys.stdout.reconfigure(encoding="utf-8")
    for line in sys.stdin:
        if not line.strip():
            continue
        msg = json.loads(line)
        if "id" not in msg:   # notification
            continue
        try:
            reply = {"jsonrpc": "2.0", "id": msg["id"], "result": handle(msg)}
        except KeyError as e:
            reply = {"jsonrpc": "2.0", "id": msg["id"], "error": {"code": -32601, "message": f"method not found: {e}"}}
        sys.stdout.write(json.dumps(reply, ensure_ascii=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
