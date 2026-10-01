"""
log_injector.py — scope-boundary logging with two modes.

Mode "regex" — fast, free, regex-based (default)
Mode "llm"   — uses LLM to instrument code, handles any pattern

Switch in config.py: LOG_INJECTION_MODE = "regex" or "llm"

Key design principle:
  Log lines are for OUR monitoring only.
  The agent NEVER sees them — they are stripped before
  the output is fed back as observation.
"""
import re

# ── Sensitive variable names — never log these ────────────────────────────────
_SKIP_VARNAMES = {
    "API_KEY", "api_key", "KEY", "key", "TOKEN", "token",
    "SECRET", "secret", "PASSWORD", "password",
    "BASE", "base", "URL", "url", "HOST", "host",
}

# ── Prefix for ALL injected log lines ─────────────────────────────────────────
# Used to strip them from agent observation
LOG_PREFIX = "[S"


def is_log_line(line: str) -> bool:
    """True if this line was injected by us — not from the agent's own code."""
    s = line.strip()
    return (s.startswith("[S2:ToolInput]") or
            s.startswith("[S3:ResponseProcessing]") or
            s.startswith("[S4:FinalAnswer]"))


def strip_log_lines(output: str) -> str:
    """
    Remove injected log lines from execution output.
    Call this before feeding output back to the agent.
    The agent should only see its own print() output.
    """
    return "\n".join(
        line for line in output.splitlines()
        if not is_log_line(line)
    ).strip()


def keep_log_lines(output: str) -> str:
    """
    Keep ONLY the injected [Sx] log lines from execution output.
    This is the 'logged output' — shown for analysis/display only,
    never fed back to the agent.
    """
    return "\n".join(
        line for line in output.splitlines()
        if is_log_line(line)
    ).strip()


# ── Regex-based injection ─────────────────────────────────────────────────────

def _skip_line(line: str) -> bool:
    """True if this line should not get an extraction log."""
    if re.search(r"os\.environ|os\.getenv", line): return True
    if re.match(r'\s*\w+\s*=\s*["\']', line):     return True  # string
    if re.match(r'\s*\w+\s*=\s*f["\']', line):    return True  # f-string
    if re.match(r'\s*\w+\s*=\s*\d', line):         return True  # number
    if re.match(r'\s*\w+\s*=\s*[\[{]', line):      return True  # list/dict
    vm = re.match(r"\s*(\w+)\s*=", line)
    if vm and vm.group(1) in _SKIP_VARNAMES:        return True
    return False


def inject_logs_regex(code: str) -> str:
    """Inject scope logs using regex pattern matching."""
    if not code or not code.strip():
        return code

    lines = code.split("\n")
    out   = []

    for line in lines:
        stripped = line.strip()
        pad      = " " * (len(line) - len(line.lstrip()))

        if not stripped or stripped.startswith("#"):
            out.append(line)
            continue

        # S2: before any requests call — print method + URL hint
        if re.search(r"\brequests\.(get|post|put|delete|patch)\b", line):
            url_m    = re.search(r'["\']([^"\']{4,})["\']', line)
            url_hint = url_m.group(1)[-80:] if url_m else "?"
            meth_m   = re.search(r"requests\.(\w+)", line)
            method   = meth_m.group(1).upper() if meth_m else "CALL"
            out.append(f'{pad}print("[S2:ToolInput] {method} {url_hint}")')
            out.append(line)
            # S3: print stored response variable (only if not chained .json())
            vm = re.match(r"\s*(\w+)\s*=\s*requests\.", line)
            if vm and ".json()" not in line:
                out.append(f'{pad}print("[S3:ResponseProcessing] {vm.group(1)} =", str({vm.group(1)})[:300])')
            continue

        # S3: variable extracted by indexing (e.g. person_id = data["results"][0]["id"])
        if (re.search(r'^\s*\w+\s*=\s*.+(?:\[.+?\])+', line) and
                not re.search(r"\brequests\.", line) and
                not stripped.startswith(("print", "for ", "if ", "return",
                                         "while ", "try:", "except", "with ")) and
                not _skip_line(line)):
            vm = re.match(r"\s*(\w+)\s*=", line)
            if vm:
                vname = vm.group(1)
                out.append(line)
                out.append(f'{pad}print("[S3:ResponseProcessing] {vname} =", str({vname})[:200])')
                continue

        # S4: before final answer print
        if "print" in line and re.search(r"FINAL ANSWER:|Answer:", line):
            am = re.search(r'print\s*\(\s*["\'][^"\']*["\'],?\s*(.+?)\s*\)$', line)
            if am:
                avar = am.group(1).strip()
                if avar and not avar.startswith(('"', "'")):
                    out.append(f'{pad}print("[S4:FinalAnswer] type:", type({avar}).__name__, "| value:", str({avar})[:200])')
            out.append(line)
            continue

        out.append(line)

    return "\n".join(out)


# ── AST-based injection (accurate static analysis) ───────────────────────────
# Parses code into a syntax tree, so it correctly identifies assignments,
# requests calls, and extractions regardless of formatting / line breaks /
# chained calls. GUARANTEE: only inserts print() statements — never alters
# existing logic (semantics are preserved by ast.unparse).

import ast as _ast

_HTTP_METHODS = {"get", "post", "put", "delete", "patch", "head", "options"}


def _find_requests_call(node):
    """Return the requests.<method>(...) Call node inside `node`, or None."""
    for n in _ast.walk(node):
        if (isinstance(n, _ast.Call) and isinstance(n.func, _ast.Attribute)
                and isinstance(n.func.value, _ast.Name)
                and n.func.value.id == "requests"
                and n.func.attr in _HTTP_METHODS):
            return n
    return None


def _unparse_url(node):
    """Reconstruct a URL string from a literal, an f-string, or a concatenation.
    f-string placeholders become {var} so the endpoint path is still visible."""
    if isinstance(node, _ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, _ast.JoinedStr):          # f-string
        parts = []
        for v in node.values:
            if isinstance(v, _ast.Constant) and isinstance(v.value, str):
                parts.append(v.value)
            elif isinstance(v, _ast.FormattedValue):
                # represent the interpolated expression compactly as {name/expr}
                try:
                    parts.append("{" + _ast.unparse(v.value) + "}")
                except Exception:
                    parts.append("{}")
        return "".join(parts)
    if isinstance(node, _ast.BinOp) and isinstance(node.op, _ast.Add):  # "a" + "b"
        left  = _unparse_url(node.left)
        right = _unparse_url(node.right)
        if left is not None and right is not None:
            return left + right
    return None


def _url_hint(call):
    """Best-effort URL string from a requests call (literal, f-string, or concat)."""
    if call.args:
        u = _unparse_url(call.args[0])
        if u:
            return u[-80:]
    for kw in call.keywords:
        if kw.arg == "url":
            u = _unparse_url(kw.value)
            if u:
                return u[-80:]
    return "?"


def _assign_target_name(stmt):
    """Single simple Name target of an assignment, else None."""
    if isinstance(stmt, _ast.Assign) and len(stmt.targets) == 1 \
            and isinstance(stmt.targets[0], _ast.Name):
        return stmt.targets[0].id
    if isinstance(stmt, _ast.AnnAssign) and isinstance(stmt.target, _ast.Name):
        return stmt.target.id
    return None


def _is_extraction(value):
    """True if value reads from an existing structure (subscript), not a request."""
    if _find_requests_call(value):
        return False
    return any(isinstance(n, _ast.Subscript) for n in _ast.walk(value))


def _has_json_call(value):
    """True if value contains a .json() method call (parsed response body)."""
    for n in _ast.walk(value):
        if (isinstance(n, _ast.Call) and isinstance(n.func, _ast.Attribute)
                and n.func.attr == "json"):
            return True
    return False


def _stmt(src):
    """Parse a one-line statement string into an AST node."""
    return _ast.parse(src).body[0]


def _instrument_body(body):
    """Return a new statement list with [Sx] prints inserted."""
    new = []
    for stmt in body:
        # Recurse into nested blocks first (for / while / if / with / try)
        for field in ("body", "orelse", "finalbody"):
            sub = getattr(stmt, field, None)
            if isinstance(sub, list):
                setattr(stmt, field, _instrument_body(sub))
        if isinstance(stmt, _ast.Try):
            for h in stmt.handlers:
                h.body = _instrument_body(h.body)

        var   = _assign_target_name(stmt)
        value = getattr(stmt, "value", None)

        # ── S2 + S3: assignment whose value makes a requests call ─────
        if value is not None and var and var not in _SKIP_VARNAMES:
            call = _find_requests_call(value)
            if call is not None:
                method = call.func.attr.upper()
                hint   = _url_hint(call).replace('"', "'")
                if hint == "?":
                    # URL wasn't a literal/f-string in the call. If it's a variable
                    # (e.g. requests.get(search_url, ...)), print that variable's
                    # runtime value so the real endpoint is captured.
                    url_arg = call.args[0] if call.args else next(
                        (kw.value for kw in call.keywords if kw.arg == "url"), None)
                    if isinstance(url_arg, _ast.Name) and url_arg.id not in _SKIP_VARNAMES:
                        new.append(_stmt(
                            f'print("[S2:ToolInput] {method}", {url_arg.id})'))
                    else:
                        new.append(_stmt(f'print("[S2:ToolInput] {method} ?")'))
                else:
                    new.append(_stmt(f'print("[S2:ToolInput] {method} {hint}")'))
                new.append(stmt)
                # If chained with .json() the var holds PARSED data (meaningful);
                # otherwise it holds the Response object (status signal).
                if _has_json_call(value):
                    new.append(_stmt(f'print("[S3:ResponseProcessing] {var} =", str({var})[:400])'))
                else:
                    new.append(_stmt(f'print("[S3:ResponseProcessing] {var} (response) =", str({var})[:120])'))
                continue

            # ── S3: parsed response body via .json() — the KEY state ──────
            if _has_json_call(value):
                new.append(stmt)
                new.append(_stmt(f'print("[S3:ResponseProcessing] {var} =", str({var})[:400])'))
                continue

            # ── S3: value extracted from an existing structure ────────
            if _is_extraction(value):
                # Don't log reads from os.environ / obvious secret sources.
                src = ""
                try:
                    src = _ast.unparse(value)
                except Exception:
                    pass
                if "environ" in src or "getenv" in src or "api_key" in src.lower():
                    new.append(stmt)
                    continue
                new.append(stmt)
                new.append(_stmt(f'print("[S3:ResponseProcessing] {var} =", str({var})[:200])'))
                continue

        # ── S4: final answer print ────────────────────────────────────
        if isinstance(stmt, _ast.Expr) and isinstance(stmt.value, _ast.Call) \
                and isinstance(stmt.value.func, _ast.Name) \
                and stmt.value.func.id == "print":
            args = stmt.value.args
            has_marker = any(
                isinstance(a, _ast.Constant) and isinstance(a.value, str)
                and ("FINAL ANSWER" in a.value or "Answer:" in a.value)
                for a in args
            )
            name_arg = next((a for a in args if isinstance(a, _ast.Name)), None)
            if has_marker and name_arg is not None \
                    and name_arg.id not in _SKIP_VARNAMES:
                new.append(_stmt(
                    f'print("[S4:FinalAnswer] type:", type({name_arg.id}).__name__, '
                    f'"| value:", str({name_arg.id})[:200])'))
                new.append(stmt)
                continue

        new.append(stmt)
    return new


def inject_logs_ast(code: str) -> str:
    """Inject scope logs using AST analysis — accurate, additive, deterministic."""
    if not code or not code.strip():
        return code
    try:
        tree = _ast.parse(code)
    except SyntaxError:
        # Unparseable (e.g. partial snippet) — fall back to regex
        return inject_logs_regex(code)
    tree.body = _instrument_body(tree.body)
    _ast.fix_missing_locations(tree)
    return _ast.unparse(tree)


# ── LLM-based injection ───────────────────────────────────────────────────────

_LLM_PROMPT = """You are a code instrumentation tool. Add print() statements to the Python
code below at exactly these three points. Do NOT change any existing logic, variable
names, or output — only ADD print() lines.

1. BEFORE every requests.get/post/put/delete(...) call, add a line that prints the HTTP
   method and the ACTUAL url being requested (interpolate the real url expression — do
   NOT print the placeholder text). For example, if the code calls
       resp = requests.get(search_url, params=params)
   insert immediately before it:
       print(f"[S2:ToolInput] GET {search_url}")
   and if the code calls
       requests.get(f"{base}/movie/{mid}/credits", params=p)
   insert:
       print(f"[S2:ToolInput] GET {base}/movie/{mid}/credits")

2. AFTER every line that assigns an API response or a value extracted from one, add:
       print("[S3:ResponseProcessing] <varname> =", str(<varname>)[:300])
   replacing <varname> with the actual variable that was just assigned.

3. BEFORE the line that prints/produces the final answer, add:
       print("[S4:FinalAnswer] type:", type(<var>).__name__, "| value:", str(<var>)[:200])
   replacing <var> with the actual answer variable/expression.

Rules:
- Interpolate REAL variable names and url expressions; never emit the literal words
  "url_hint", "varname", "METHOD", or "<var>".
- Do NOT log API keys, tokens, secrets, base URLs, or config variables.
- Do NOT change existing logic, variable names, or existing print() output.
- Return ONLY the instrumented Python code inside a ```python code block.

Code to instrument:
{code}"""


def inject_logs_llm(code: str, client, model: str) -> str:
    """Inject scope logs using an LLM — handles any code pattern.
    Prints which path actually ran so LLM-mode is not silently indistinguishable
    from AST-mode (it falls back to AST on error or unchanged output)."""
    if not code or not code.strip():
        return code

    from utils.executor import extract_code_blocks

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user",
                       "content": _LLM_PROMPT.format(code=code)}],
            temperature=0.0,
        )
        result = response.choices[0].message.content
        instrumented = extract_code_blocks(result)
        if not instrumented or instrumented.strip() == code.strip():
            print("[inject] LLM returned unchanged/empty code → fell back to AST")
            return inject_logs_ast(code)
        print("[inject] LLM injection applied")
        return instrumented
    except Exception as e:
        print(f"[inject] LLM call FAILED ({str(e)[:80]}) → fell back to AST")
        return inject_logs_ast(code)


# ── Public entry point ────────────────────────────────────────────────────────

def inject_logs(code: str, client=None, model: str = None) -> str:
    """
    Prepare the code to run for the mode set in config.LOG_INJECTION_MODE:
      "wrapper" — NO code modification (default). Logging is done at the library
                  boundary by utils.exec_wrapper inside the interpreter, so the
                  agent's code runs verbatim. This returns code unchanged.
      "ast"     — accurate static analysis: inserts print() scope logs.
      "regex"   — legacy pattern matching (kept for comparison).
      "llm"     — LLM-based instrumentation (needs client + model).
    LLM mode falls back to AST if client/model are missing or the call fails.
    """
    import config
    mode = getattr(config, "LOG_INJECTION_MODE", "wrapper")

    if mode == "wrapper":
        return code                      # unmodified — wrapper logs at runtime
    elif mode == "llm" and client is not None and model is not None:
        return inject_logs_llm(code, client, model)
    elif mode == "regex":
        return inject_logs_regex(code)
    elif mode == "ast":
        return inject_logs_ast(code)
    else:  # unrecognized → safest faithful default (no code modification)
        return code
