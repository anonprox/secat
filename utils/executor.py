"""
executor.py — PersistentInterpreter for SECAT agents.

Each execute() call runs in a subprocess for full isolation:
  - CPU-spinning loops are killed by timeout (no GIL problem)
  - A crash in agent code never kills the runner
  - Namespace is persisted between turns via pickle files
"""
import io, os, sys, json, pickle, subprocess, tempfile, traceback, textwrap

TIMEOUT = 30
_MISSING = object()

def _get_timeout():
    try:
        import config as _cfg
        return int(getattr(_cfg, "CODE_TIMEOUT_SEC", TIMEOUT))
    except Exception:
        return TIMEOUT


# The runner script that executes inside the subprocess.
# Written to a file — avoids f-string / escaping issues.
_RUNNER_TEMPLATE = '''\
import pickle, sys, os, traceback

NS_IN  = sys.argv[1]
NS_OUT = sys.argv[2]
CODE_F = sys.argv[3]
SCOPE_F= sys.argv[4]

with open(NS_IN, "rb") as _f:
    _ns = pickle.load(_f)

import requests as _real_requests
import json, re, math, collections
_project_root = os.environ.get("SECAT_PROJECT_ROOT")
if _project_root and _project_root not in sys.path:
    sys.path.insert(0, _project_root)
_runtime_api = os.environ.get("SECAT_BENCHMARK")
if _runtime_api:
    from utils.api_runtime import install_requests_compat as _secat_install_requests
    try:
        from utils.api_runtime import credential_env_key as _secat_credential_env_key
        _secat_secret_env = _secat_credential_env_key(_runtime_api)
    except Exception:
        _secat_secret_env = ""
    _real_requests = _secat_install_requests(_runtime_api)
else:
    _secat_secret_env = ""
# The compatibility transport is imported by the trusted runner.  Agent code
# then executes from an empty workspace without the project/gold tree on its
# import path.  This is evaluation isolation, not an OS security sandbox.
if os.environ.get("SECAT_EVALUATION_PROFILE", "isolated") == "isolated":
    os.environ.pop("SECAT_PROJECT_ROOT", None)
    os.environ.pop("SECAT_BENCHMARK", None)
    if _secat_secret_env:
        os.environ.pop(_secat_secret_env, None)
    if _project_root:
        sys.path = [p for p in sys.path if os.path.abspath(p or os.curdir) != os.path.abspath(_project_root)]
_ns.update({
    "json": json, "re": re, "os": os,
    "math": math, "collections": collections,
    "__builtins__": __builtins__,
})

# Lightweight scope logger for [S2]/[S3] annotation lines
_scope_lines = []

class _LogReq:
    """Thin wrapper that logs HTTP calls then delegates to real requests."""
    @staticmethod
    def _log(method, url, **kw):
        _scope_lines.append(f"[S2] HTTP {method.upper()} {url}")
        fn = getattr(_real_requests, method)
        resp = fn(url, **kw)
        _scope_lines.append(f"[S3] HTTP {resp.status_code}")
        return resp
    request = staticmethod(lambda method, url, **kw: _LogReq._log(method, url, **kw))
    get     = staticmethod(lambda url, **kw: _LogReq._log("get",  url, **kw))
    post    = staticmethod(lambda url, **kw: _LogReq._log("post", url, **kw))
    put     = staticmethod(lambda url, **kw: _LogReq._log("put", url, **kw))
    delete  = staticmethod(lambda url, **kw: _LogReq._log("delete", url, **kw))
    patch   = staticmethod(lambda url, **kw: _LogReq._log("patch", url, **kw))
    Session = _real_requests.Session
    codes   = _real_requests.codes
    exceptions = _real_requests.exceptions

_requests_proxy = _LogReq()
_ns["requests"] = _requests_proxy
sys.modules["requests"] = _requests_proxy

_agent_code = open(CODE_F).read()
_ok = True
try:
    exec(compile(_agent_code, "<agent_code>", "exec"), _ns)
except SystemExit:
    pass
except Exception:
    print(traceback.format_exc(), file=sys.stderr)
    _ok = False

# Save namespace (picklable values only, skip private/module entries)
_save = {}
for _k, _v in _ns.items():
    if _k.startswith("_") or _k in ("__builtins__",):
        continue
    # skip the logging wrapper — next turn will re-inject it
    if _k == "requests":
        continue
    try:
        pickle.dumps(_v)
        _save[_k] = _v
    except Exception:
        pass

with open(NS_OUT, "wb") as _f:
    pickle.dump(_save, _f)

with open(SCOPE_F, "w") as _f:
    _f.write("\\n".join(_scope_lines))

sys.exit(0 if _ok else 1)
'''


class PersistentInterpreter:
    """
    Subprocess-based interpreter. Each execute() call runs in a fresh process,
    sharing namespace state via pickle files between turns.
    """

    def __init__(self):
        self.namespace: dict = {}
        self._workspace = tempfile.TemporaryDirectory(prefix="secat_agent_workspace_")
        # Write the runner script to a persistent temp file once
        self._runner_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, prefix="secat_runner_")
        self._runner_file.write(_RUNNER_TEMPLATE)
        self._runner_file.close()
        self._owns_runner = True

    def execute(self, code: str, timeout: int = None) -> dict:
        if timeout is None:
            timeout = _get_timeout()

        result = {
            "stdout": "", "stderr": "", "combined": "",
            "exit_code": 0, "timed_out": False, "success": False,
            "var_trace": [],
        }

        if not code or not code.strip():
            result["stderr"] = result["combined"] = "No code provided"
            return result

        with tempfile.TemporaryDirectory() as tmpdir:
            ns_in   = os.path.join(tmpdir, "ns_in.pkl")
            ns_out  = os.path.join(tmpdir, "ns_out.pkl")
            code_f  = os.path.join(tmpdir, "agent_code.py")
            scope_f = os.path.join(tmpdir, "scope.txt")

            # Serialize current namespace
            picklable = {}
            for k, v in self.namespace.items():
                try:
                    pickle.dumps(v)
                    picklable[k] = v
                except Exception:
                    pass
            with open(ns_in, "wb") as f:
                pickle.dump(picklable, f)
            with open(code_f, "w") as f:
                f.write(code)

            try:
                proc = subprocess.run(
                    [sys.executable, self._runner_file.name,
                     ns_in, ns_out, code_f, scope_f],
                    capture_output=True, text=True, cwd=self._workspace.name,
                    timeout=timeout, env={**dict(os.environ),
                                          "SECAT_PROJECT_ROOT": os.path.dirname(os.path.dirname(os.path.abspath(__file__)))},
                )
                result["stdout"]    = proc.stdout
                result["stderr"]    = proc.stderr
                result["exit_code"] = proc.returncode
                result["success"]   = (proc.returncode == 0)

                # Load updated namespace
                if os.path.exists(ns_out):
                    try:
                        with open(ns_out, "rb") as f:
                            self.namespace.update(pickle.load(f))
                    except Exception:
                        pass

                # Load scope lines for [Sx] annotation
                scope_text = ""
                if os.path.exists(scope_f):
                    try:
                        with open(scope_f, encoding="utf-8") as scope_handle:
                            scope_text = scope_handle.read().strip()
                    except Exception:
                        pass

            except subprocess.TimeoutExpired:
                result["timed_out"] = True
                result["stderr"]    = f"Code execution timed out after {timeout}s"
                result["exit_code"] = -1
                scope_text = ""

        combined = result["stdout"] + result["stderr"]
        if scope_text:
            combined = combined.rstrip("\n") + "\n" + scope_text + "\n"
        result["combined"] = combined
        from utils.runtime_abort import raise_if_fatal_runtime_signal
        raise_if_fatal_runtime_signal(combined)
        return result

    def reset(self):
        """Clear namespace between tasks."""
        self.namespace = {}

    def fork(self):
        """Return an independent copy for CodeTool candidate scoring."""
        f = PersistentInterpreter.__new__(PersistentInterpreter)
        f.namespace = dict(self.namespace)
        f._runner_file = self._runner_file   # share the trusted runner script
        f._owns_runner = False
        f._workspace = tempfile.TemporaryDirectory(prefix="secat_agent_workspace_")
        return f

    def shutdown(self):
        self.reset()
        if getattr(self, "_owns_runner", False):
            try:
                os.unlink(self._runner_file.name)
            except Exception:
                pass
        try:
            self._workspace.cleanup()
        except Exception:
            pass

    def __del__(self):
        if getattr(self, "_owns_runner", False):
            try:
                os.unlink(self._runner_file.name)
            except Exception:
                pass
        try:
            self._workspace.cleanup()
        except Exception:
            pass


def extract_code_blocks(text: str) -> str:
    """Return the first ```python...``` or <execute>...</execute> block, or ''."""
    import re
    m = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"<execute>(.*?)</execute>", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""
