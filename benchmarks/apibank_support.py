"""Standalone scoped loader/execution guards for the unmodified API-Bank vendor.

ToolCoder workers may copy this file and call tool_manager_class(root, api_names).
Only requested API modules and CheckToken are imported. Calls require the main
thread on Unix because an uncatchable-by-Exception SIGALRM bounds vendor code.
"""
from contextlib import contextmanager
import ast
import asyncio
import importlib
import inspect
import os
from pathlib import Path
import signal
import sys
import threading
import time


class APIBankInfrastructureError(Exception):
    error_type = 'tool_service_error'


class APIBankToolTimeout(APIBankInfrastructureError):
    error_type = 'tool_timeout'


class APIBankDependencyError(APIBankInfrastructureError):
    error_type = 'tool_dependency_error'


class _Deadline(BaseException):
    pass


class _ServiceFailure(BaseException):
    pass


_LOCK = threading.RLock()
_MODULES = {}
_CLASS_INDEX = {}


def _vendor_name(name):
    return name in {'apis', 'tool_manager', 'api_call_extraction'} or name.startswith('apis.')


def _allowed_files(root, names):
    toolsearcher_scope = names is not None and 'ToolSearcher' in set(names)
    if names is None:
        # A diagnostic manager can load local APIs, but never needs ToolSearcher.
        names = None
    else:
        names = set(names) | {'CheckToken'}
    paths = sorted((root / 'apis').glob('*.py'))
    signature = tuple((str(path), path.stat().st_mtime_ns, path.stat().st_size) for path in paths)
    cached = _CLASS_INDEX.get(str(root))
    if cached is None or cached[0] != signature:
        classes_by_file = {}
        for path in paths:
            tree = ast.parse(path.read_text(encoding='utf-8'))
            classes_by_file[path.name] = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
        _CLASS_INDEX[str(root)] = (signature, classes_by_file)
    else:
        classes_by_file = cached[1]
    allowed = {'__init__.py', 'api.py'}
    for filename, classes in classes_by_file.items():
        # ToolSearcher is a meta-tool. Its upstream constructor scans every
        # vendor API module and explicitly requires GetUserToken while building
        # its private retrieval index. This broad import scope is execution-only;
        # model-visible descriptions stay progressively scoped by the runtime.
        if toolsearcher_scope:
            allowed.add(filename)
        elif (names is None and 'ToolSearcher' not in classes) or (names is not None and classes & names):
            allowed.add(filename)
    return allowed


@contextmanager
def vendor_context(root, api_names=None):
    """Temporarily activate a root's imports; never leak them into another root."""
    root = Path(root).resolve()
    with _LOCK:
        old_cwd, old_path, old_env = Path.cwd(), list(sys.path), dict(os.environ)
        allowed = _allowed_files(root, api_names)
        previous = {k:v for k,v in sys.modules.items() if _vendor_name(k)}
        for name in previous:
            sys.modules.pop(name, None)
        cache = _MODULES.setdefault(str(root), {})
        sys.modules.update(cache)
        original_listdir = os.listdir
        def scoped_listdir(path='.'):
            entries = original_listdir(path)
            if not isinstance(path, int) and Path(path).resolve() == root / 'apis':
                return [entry for entry in entries if entry in allowed or not entry.endswith('.py')]
            return entries
        sys.path.insert(0, str(root))
        os.chdir(root)
        os.listdir = scoped_listdir
        try:
            yield
        finally:
            os.listdir = original_listdir
            cache.update({k:v for k,v in sys.modules.items() if _vendor_name(k)})
            for name in list(sys.modules):
                if _vendor_name(name):
                    sys.modules.pop(name, None)
            sys.modules.update(previous)
            os.chdir(old_cwd)
            sys.path[:] = old_path
            os.environ.clear()
            os.environ.update(old_env)


def load_vendor_module(root, name):
    with vendor_context(root, ()):
        return importlib.import_module(name)


def tool_timeout_seconds():
    try:
        value = float(os.environ.get('APIBANK_TOOL_TIMEOUT_SEC', '30'))
    except (TypeError, ValueError) as exc:
        raise APIBankDependencyError('APIBANK_TOOL_TIMEOUT_SEC must be positive') from exc
    if not 0 < value < float('inf'):
        raise APIBankDependencyError('APIBANK_TOOL_TIMEOUT_SEC must be positive and finite')
    return value


@contextmanager
def execution_guard():
    """Bound tools and distinguish swallowed transport failures from bad actions."""
    if threading.current_thread() is not threading.main_thread() or not hasattr(signal, 'setitimer'):
        raise APIBankDependencyError('API-Bank tool execution requires the Unix main thread')
    seconds = tool_timeout_seconds()
    old_handler = signal.getsignal(signal.SIGALRM)
    old_timer = signal.getitimer(signal.ITIMER_REAL)
    started = time.monotonic()
    def alarm(*args):
        raise _Deadline()
    # requests is optional for purely synthetic/local runtimes.
    request_module = None
    try:
        import requests
        request_module = requests
    except ImportError:
        pass
    original_request = request_module.sessions.Session.request if request_module else None
    if request_module:
        def bounded_request(self, *args, **kwargs):
            kwargs.setdefault('timeout', seconds)
            try:
                response = original_request(self, *args, **kwargs)
            except request_module.exceptions.RequestException as exc:
                raise _ServiceFailure(str(exc)) from exc
            if response.status_code in {401, 403, 429} or response.status_code >= 500:
                raise _ServiceFailure('HTTP service failure: %s' % response.status_code)
            return response
        request_module.sessions.Session.request = bounded_request
    signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, min(seconds, old_timer[0]) if old_timer[0] else seconds)
    try:
        yield
    except _Deadline as exc:
        raise APIBankToolTimeout('API-Bank tool exceeded %.3g seconds' % seconds) from exc
    except _ServiceFailure as exc:
        raise APIBankInfrastructureError(str(exc)) from exc
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        if old_timer[0]:
            signal.setitimer(signal.ITIMER_REAL, max(0.000001, old_timer[0] - (time.monotonic()-started)), old_timer[1])
        if request_module:
            request_module.sessions.Session.request = original_request


class _SynchronousTranslator:
    """Run a tool's async googletrans methods on one explicitly owned loop."""
    def __init__(self, translator):
        self.translator = translator
        self.translator_type = type(translator)
        self.loop = None
        self.closed = False

    def reset(self):
        if self.closed:
            # The vendor constructs Translator() with no customization. Reset is
            # called before its call() can mutate proxy environment variables.
            self.translator = self.translator_type()
            self.closed = False

    def _invoke(self, name, *args, **kwargs):
        try:
            method = getattr(self.translator, name)
            if inspect.iscoroutinefunction(method):
                if self.loop is None:
                    self.loop = asyncio.new_event_loop()
                result = self.loop.run_until_complete(method(*args, **kwargs))
            else:
                result = method(*args, **kwargs)
            response = getattr(result, '_response', None)
            status = getattr(response, 'status_code', 200)
            if status in {401, 403, 429} or status >= 500:
                raise _ServiceFailure('Translation service failure: %s' % status)
            return result
        except Exception as exc:
            import httpx
            if isinstance(exc, (httpx.HTTPError, OSError)):
                raise _ServiceFailure(str(exc)) from exc
            raise

    def close(self):
        if self.closed:
            return
        client = getattr(self.translator, 'client', None)
        try:
            if client is not None and hasattr(client, 'aclose'):
                if self.loop is None:
                    self.loop = asyncio.new_event_loop()
                self.loop.run_until_complete(client.aclose())
        finally:
            if self.loop is not None:
                self.loop.close()
                self.loop = None
            self.closed = True

    def translate(self, *args, **kwargs):
        return self._invoke('translate', *args, **kwargs)

    def detect(self, *args, **kwargs):
        return self._invoke('detect', *args, **kwargs)


def normalize_parameters(manager, name, params):
    values = dict(params or {})
    getter = getattr(manager, 'get_api_by_name', None)
    schema = getter(name).get('input_parameters', {}) if callable(getter) else {}
    for key, value in values.items():
        if isinstance(value, bool) and schema.get(key, {}).get('type') == 'bool':
            values[key] = 'True' if value else 'False'
    return values


def tool_manager_class(root, api_names=None, *, base_class=None):
    """Return an upstream ToolManager subclass with scoped imports and guards."""
    root = Path(root).resolve()
    names = tuple(api_names) if api_names is not None else None
    if base_class is None:
        base_class = load_vendor_module(root, 'tool_manager').ToolManager
    class ScopedToolManager(base_class):
        def __init__(self, *args, **kwargs):
            with vendor_context(root, names):
                super().__init__(*args, **kwargs)

        def init_tool(self, name, *args, **kwargs):
            with vendor_context(root, names):
                tool = super().init_tool(name, *args, **kwargs)
                if name == 'Translate' and hasattr(tool, 'translator') and not isinstance(tool.translator, _SynchronousTranslator):
                    tool.translator = _SynchronousTranslator(tool.translator)
                if name == 'Translate' and isinstance(getattr(tool, 'translator', None), _SynchronousTranslator):
                    tool.translator.reset()
                return tool

        def close(self):
            for tool in getattr(self, 'inited_tools', {}).values():
                translator = getattr(tool, 'translator', None)
                if isinstance(translator, _SynchronousTranslator):
                    translator.close()

        def api_call(self, tool_name=None, *args, **kwargs):
            with vendor_context(root, names), execution_guard():
                try:
                    return super().api_call(tool_name, *args, **normalize_parameters(self, tool_name, kwargs))
                finally:
                    self.close()
    return ScopedToolManager
