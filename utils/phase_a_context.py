"""Safe deterministic compaction of OCA Phase-A chat history."""
from __future__ import annotations

import ast
import re


def _assigned_names(code: str) -> list[str]:
    try:
        tree = ast.parse(code or "")
    except Exception:
        return []
    names: list[str] = []

    def add_target(node):
        if isinstance(node, ast.Name):
            names.append(node.id)
        elif isinstance(node, (ast.Tuple, ast.List)):
            for item in node.elts:
                add_target(item)

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                add_target(t)
        elif isinstance(node, ast.AnnAssign):
            add_target(node.target)
        elif isinstance(node, (ast.For, ast.comprehension)):
            add_target(node.target)
    return list(dict.fromkeys(names))[:20]


def _code_digest(text: str) -> str:
    match = re.search(r"<execute>(.*?)</execute>", text or "", re.S)
    if not match:
        short = " ".join((text or "").split())
        return short[:500]
    code = match.group(1).strip()
    names = _assigned_names(code)
    methods = re.findall(r"requests\.(get|post|put|delete|patch)\s*\(", code, re.I)
    endpoints = re.findall(r"['\"](https?://[^'\"]+|/[^'\"]+)['\"]", code)
    bits = ["Prior code executed in persistent kernel."]
    if names:
        bits.append("variables=" + ",".join(names))
    if methods:
        bits.append("http=" + ",".join(m.upper() for m in methods[:8]))
    if endpoints:
        bits.append("paths=" + ",".join(x[:120] for x in endpoints[:8]))
    return " ".join(bits)[:900]


def compact_messages(messages: list[dict], keep_recent_pairs: int = 1,
                     old_feedback_max_chars: int = 900) -> list[dict]:
    """Return a compact *view* of a Phase-A transcript.

    The original message list is not mutated. The system prompt and original task
    are preserved exactly, the most recent execution pair remains exact, and older
    assistant code is replaced by a deterministic variable/endpoint digest. Older
    observation receipts are already compact and are bounded once more here.
    """
    if len(messages) <= 4:
        return [dict(m) for m in messages]
    out = [dict(messages[0]), dict(messages[1])]
    tail_count = max(0, int(keep_recent_pairs)) * 2
    cutoff = max(2, len(messages) - tail_count)
    for i, msg in enumerate(messages[2:], start=2):
        item = dict(msg)
        if i >= cutoff:
            out.append(item)
            continue
        if item.get("role") == "assistant":
            item["content"] = _code_digest(str(item.get("content") or ""))
        elif item.get("role") == "user":
            text = str(item.get("content") or "")
            if len(text) > old_feedback_max_chars:
                text = text[:old_feedback_max_chars] + "\n… older receipt compacted"
            item["content"] = text
        out.append(item)
    return out
