"""
kernel_executor.py
A CodeAct execution backend that runs agent code in a REAL IPython/Jupyter kernel,
exactly like the original CodeAct (Wang et al. 2024). The defining difference from
the script-based PersistentInterpreter: a trailing BARE EXPRESSION is auto-displayed
by the kernel (Out[n]: ...), so the agent observes its result without an explicit
print(). This is precisely what eliminates the S4 "perception" failure.

Same interface as PersistentInterpreter:
    interp = KernelInterpreter()
    res = interp.execute(code)   # -> {stdout, stderr, combined, success, ...}
    interp.shutdown()

Variables persist across execute() calls (the kernel keeps state), matching CodeAct.

Requires: pip install jupyter_client ipykernel
We capture the kernel's stdout, any errors, AND the auto-displayed execute_result /
display_data (the trailing-expression value). That auto-display is the key signal:
in script mode it is absent; here it is present.

NOTE: the requests-wrapper + sys.settrace tracer used in script mode run in THIS
process; they do not reach into the kernel subprocess. So kernel mode trades the
fine-grained [S2]/[S3] variable capture for fidelity to CodeAct's observation model.
For the S4 comparison that is the right trade: what matters is whether the trailing
expression is observed, and the kernel's execute_result captures exactly that.
"""
import os
import queue
import tempfile
import json
import re


def python_json_assignment(name: str, value, json_alias: str = "_json") -> str:
    """Build Python source that reconstructs JSON data safely inside the kernel.

    Never paste JSON directly as a Python literal: JSON ``null``/``true``/``false``
    are not Python names.  Encoding the payload as a quoted string and parsing it
    inside the trusted kernel also handles arbitrary user/API strings safely.
    """
    if not re.fullmatch(r"[A-Za-z_]\w*", str(name or "")):
        raise ValueError(f"invalid Python assignment target: {name!r}")
    if not re.fullmatch(r"[A-Za-z_]\w*", str(json_alias or "")):
        raise ValueError(f"invalid JSON module alias: {json_alias!r}")
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    # Re-import the module in the same trusted statement.  Agent/generated code
    # runs in a persistent kernel and may legally reuse names such as ``_json``
    # for ordinary data.  Relying on a bootstrap-time alias therefore lets model
    # state corrupt later host control-plane updates (e.g. ``_json`` becomes a
    # dict and ``_json.loads`` crashes).  Rebinding the alias here makes each
    # trusted assignment self-contained without resetting the user namespace.
    return (f"import json as {json_alias}\n"
            f"{name} = {json_alias}.loads({payload!r})")



class KernelInterpreter:
    def __init__(self, startup_timeout: int = 30):
        import sys
        from jupyter_client import KernelManager
        # Launch a kernel using the CURRENT Python interpreter directly, rather
        # than relying on a kernelspec named "python3" being registered (it often
        # is NOT, e.g. in a fresh conda env -> "No such kernel named python3").
        # Setting kernel_cmd makes the manager run *this* interpreter as the kernel.
        self._workspace = tempfile.TemporaryDirectory(prefix="secat_agent_workspace_")
        self.km = KernelManager()
        try:
            self.km.kernel_cmd = [sys.executable, "-m", "ipykernel_launcher",
                                  "-f", "{connection_file}"]
            # kernel_cmd bypasses kernelspec; silence the related deprecation path
            try:
                self.km.start_kernel(cwd=self._workspace.name)
            except Exception:
                # last-resort fallbacks: a registered "python3", then the default spec
                self.km = KernelManager(kernel_name="python3")
                try:
                    self.km.start_kernel(cwd=self._workspace.name)
                except Exception:
                    self.km = KernelManager()
                    self.km.start_kernel(cwd=self._workspace.name)
        except Exception:
            raise
        self.kc = self.km.client()
        self.kc.start_channels()
        try:
            self.kc.wait_for_ready(timeout=startup_timeout)
        except RuntimeError:
            self.shutdown()
            raise
        # Pre-import common libs, then install the registry-configured requests
        # transport *before* evaluation isolation removes the project source tree
        # from the kernel import path.  This is intentionally API-agnostic: bearer,
        # query-key, refresh-token, or future transport details live in the registry
        # adapter rather than generated code or this executor.
        self._run_silent("import os, requests, json, re", check=True)
        runtime_api = os.environ.get("SECAT_BENCHMARK")
        project_root = os.environ.get("SECAT_PROJECT_ROOT")
        if runtime_api:
            startup = "import sys, os\n"
            if project_root:
                startup += f"sys.path.insert(0, {project_root!r})\n"
            startup += (
                "from utils.api_runtime import install_requests_compat as _secat_install_requests\n"
                "try:\n"
                "    from utils.api_runtime import credential_env_key as _secat_credential_env_key\n"
                f"    _secat_secret_env = _secat_credential_env_key({runtime_api!r})\n"
                "except Exception:\n"
                "    _secat_secret_env = ''\n"
                f"requests = _secat_install_requests({runtime_api!r})\n"
                "sys.modules['requests'] = requests\n"
            )
            if os.environ.get("SECAT_EVALUATION_PROFILE", "isolated") == "isolated":
                startup += (
                    "os.environ.pop('SECAT_PROJECT_ROOT', None)\n"
                    "os.environ.pop('SECAT_BENCHMARK', None)\n"
                    "if _secat_secret_env: os.environ.pop(_secat_secret_env, None)\n"
                )
                # TMDB baseline compatibility: historical CodeAct-style generated
                # code may still read TMDB_API_KEY and pass it as a query/header
                # credential.  The real secret remains captured only by the trusted
                # requests transport.  Expose a non-secret sentinel and strip that
                # sentinel from outgoing generated-code auth fields so the trusted
                # transport can apply the real credential itself.
                if runtime_api in {"tmdb", "tmdb_verified"}:
                    startup += (
                        "_secat_auth_sentinel = '__SECAT_RUNTIME_AUTH__'\n"
                        "if _secat_secret_env: os.environ[_secat_secret_env] = _secat_auth_sentinel\n"
                        "def _secat_clean_auth_kwargs(_kw, _sentinel=_secat_auth_sentinel):\n"
                        "    _kw = dict(_kw)\n"
                        "    _params = _kw.get('params')\n"
                        "    if isinstance(_params, dict):\n"
                        "        _params = dict(_params)\n"
                        "        if _params.get('api_key') == _sentinel:\n"
                        "            _params.pop('api_key', None)\n"
                        "        _kw['params'] = _params\n"
                        "    _headers = _kw.get('headers')\n"
                        "    if isinstance(_headers, dict):\n"
                        "        _headers = dict(_headers)\n"
                        "        _auth = str(_headers.get('Authorization') or '')\n"
                        "        if _sentinel in _auth:\n"
                        "            _headers.pop('Authorization', None)\n"
                        "        _kw['headers'] = _headers\n"
                        "    return _kw\n"
                        "def _secat_wrap_request_method(_fn, _clean=_secat_clean_auth_kwargs):\n"
                        "    def _wrapped(*_a, **_kw):\n"
                        "        return _fn(*_a, **_clean(_kw))\n"
                        "    return _wrapped\n"
                        "for _secat_method in ('request', 'get', 'post', 'put', 'patch', 'delete', 'head', 'options'):\n"
                        "    _secat_fn = getattr(requests, _secat_method, None)\n"
                        "    if callable(_secat_fn):\n"
                        "        setattr(requests, _secat_method, _secat_wrap_request_method(_secat_fn))\n"
                        "globals().pop('_secat_method', None)\n"
                        "globals().pop('_secat_fn', None)\n"
                        "globals().pop('_secat_auth_sentinel', None)\n"
                        "globals().pop('_secat_clean_auth_kwargs', None)\n"
                        "globals().pop('_secat_wrap_request_method', None)\n"
                    )
                startup += f"sys.path = [p for p in sys.path if p != {project_root!r}]\n"
            # Trusted bootstrap helpers are not part of the agent API.  Remove
            # them after the requests-compatible transport has captured its
            # configuration/credential so generated code cannot call or inspect
            # those helper objects later.
            startup += (
                "for _secat_name in ('_secat_install_requests', '_secat_credential_env_key', "
                "'_secat_secret_env'):\n"
                "    globals().pop(_secat_name, None)\n"
                "globals().pop('_secat_name', None)\n"
            )
            self._run_silent(startup, check=True)
        elif os.environ.get("SECAT_EVALUATION_PROFILE", "isolated") == "isolated":
            self._run_silent("os.environ.pop('SECAT_PROJECT_ROOT', None)", check=True)

    def _run_silent(self, code, check: bool = False):
        msg_id = self.kc.execute(code, store_history=False)
        drained = self._drain(msg_id, timeout=20)
        if check and drained[-1] != "ok":
            errors = drained[3] or []
            detail = "\n".join(errors) if errors else "kernel startup code failed"
            raise RuntimeError(detail)
        return drained

    def _drain(self, msg_id, timeout):
        """Collect all output messages for one execution until idle."""
        stdout, stderr, results, errors = [], [], [], []
        status = "ok"
        while True:
            try:
                msg = self.kc.get_iopub_msg(timeout=timeout)
            except queue.Empty:
                break
            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue
            mtype = msg["msg_type"]
            content = msg["content"]
            if mtype == "stream":
                (stdout if content.get("name") == "stdout" else stderr).append(content.get("text", ""))
            elif mtype in ("execute_result", "display_data"):
                # THIS is the auto-displayed trailing expression / rich output
                data = content.get("data", {})
                if "text/plain" in data:
                    results.append(data["text/plain"])
            elif mtype == "error":
                status = "error"
                errors.append("\n".join(content.get("traceback", [])))
            elif mtype == "status" and content.get("execution_state") == "idle":
                break
        return stdout, stderr, results, errors, status

    def execute(self, code: str, timeout: int = 30) -> dict:
        result = {"stdout": "", "stderr": "", "combined": "", "exit_code": 0,
                  "timed_out": False, "success": False, "var_trace": [],
                  "auto_display": ""}
        if not code or not code.strip():
            result["stderr"] = "No code provided"
            result["combined"] = result["stderr"]
            return result

        # READY_TO_ANSWER is an agent/framework protocol marker, not Python.
        # Some models place it as a bare final line inside <execute>...</execute>.
        # Remove only a standalone marker; ordinary Python is unchanged.
        import re as _secat_re
        code = _secat_re.sub(r"(?m)^\s*READY_TO_ANSWER\s*;?\s*$", "", code)

        msg_id = self.kc.execute(code, store_history=True)
        stdout, stderr, results, errors, status = self._drain(msg_id, timeout)

        out = "".join(stdout)
        err = "".join(stderr) + ("\n".join(errors) if errors else "")
        auto = "\n".join(results)  # the trailing-expression value(s), kernel-displayed

        result["stdout"] = out
        result["stderr"] = err
        result["auto_display"] = auto
        result["success"] = (status == "ok")
        result["exit_code"] = 0 if status == "ok" else 1

        # combined output the agent observes: stdout PLUS the auto-displayed value
        # (this is the crucial difference from script mode, where `auto` is empty)
        parts = []
        if out.strip():
            parts.append(out.rstrip())
        if auto.strip():
            parts.append(f"Out: {auto.rstrip()}")
        if err.strip():
            parts.append(err.rstrip())
        result["combined"] = "\n".join(parts) if parts else ""
        from utils.runtime_abort import raise_if_fatal_runtime_signal
        raise_if_fatal_runtime_signal(result["combined"])
        return result

    def clone_namespace(self):
        # parity stub; kernel keeps its own state, nothing to clone
        return None

    def shutdown(self):
        try:
            self.kc.stop_channels()
        except Exception:
            pass
        try:
            self.km.shutdown_kernel(now=True)
        except Exception:
            pass
        try:
            # jupyter_client can retain connection-file/socket resources after a
            # forced shutdown. Explicit cleanup keeps repeated offline/live kernel
            # runs from leaving process resources attached to the parent.
            self.km.cleanup_resources(restart=False)
        except Exception:
            pass
        try:
            self._workspace.cleanup()
        except Exception:
            pass

    # alias so callers that expect the PersistentInterpreter API (reset) work too
    def reset(self):
        self.shutdown()


if __name__ == "__main__":
    # demo: show that a trailing bare expression IS observed in kernel mode
    k = KernelInterpreter()
    print("--- bare expression (no print) ---")
    r = k.execute("x = 6 * 7\nx")
    print("combined:", repr(r["combined"]), "| auto_display:", repr(r["auto_display"]))
    print("--- persists across calls ---")
    r2 = k.execute("x + 1")
    print("combined:", repr(r2["combined"]))
    k.shutdown()
