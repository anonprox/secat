"""API-Bank Level-1/Level-2 runtime and official execution-based evaluator.

This module is intentionally independent of SECAT's HTTP/RestBench runtime.
API-Bank ships executable local tools and its own correctness predicates.  The
runtime loads the official ToolManager/parser from a vendored API-Bank tree,
creates one sample for each API turn, and evaluates predicted bracket calls by
executing them and invoking the target API's check_api_call_correctness().
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, replace
import importlib.util
import inspect
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple


_REQUIRED = (
    "tool_manager.py",
    "api_call_extraction.py",
    "apis",
    "lv1-lv2-samples/level-1-given-desc",
)


@dataclass(frozen=True)
class APIBankSample:
    task_id: str
    source_file: str
    source_api_index: int
    chat_history: Tuple[dict, ...]
    api_names: Tuple[str, ...]
    ground_truth: dict
    benchmark_level: int = 1
    visible_api_names: Tuple[str, ...] = ()

    def with_ground_truth(self, ground_truth: dict) -> "APIBankSample":
        return replace(self, ground_truth=deepcopy(ground_truth))


@dataclass(frozen=True)
class EvaluationResult:
    correct: bool
    predicted_api: Optional[str] = None
    predicted_params: Optional[dict] = None
    execution_result: Any = None
    replayed_calls: int = 0
    error_type: Optional[str] = None
    error: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "correct": bool(self.correct),
            "predicted_api": self.predicted_api,
            "predicted_params": deepcopy(self.predicted_params),
            "execution_result": deepcopy(self.execution_result),
            "replayed_calls": int(self.replayed_calls),
            "error_type": self.error_type,
            "error": self.error,
        }


def is_apibank_root(path: Path) -> bool:
    path = Path(path)
    return all((path / item).exists() for item in _REQUIRED)


def discover_apibank_root(path: Path) -> Path:
    """Return the API-Bank root below *path* or raise FileNotFoundError."""
    path = Path(path).expanduser().resolve()
    if is_apibank_root(path):
        return path
    if not path.exists():
        raise FileNotFoundError(f"API-Bank path does not exist: {path}")
    candidates = []
    for tool_manager in path.rglob("tool_manager.py"):
        candidate = tool_manager.parent
        if is_apibank_root(candidate):
            candidates.append(candidate)
    if not candidates:
        raise FileNotFoundError(
            "Could not locate an API-Bank root containing tool_manager.py, "
            "api_call_extraction.py, apis/, and lv1-lv2-samples/level-1-given-desc "
            f"under {path}")
    return sorted(candidates, key=lambda p: (len(p.parts), str(p)))[0]


from benchmarks.apibank_support import (
    vendor_context as _vendor_context, load_vendor_module, tool_manager_class,
    APIBankInfrastructureError, execution_guard,
)


def _load_module(path: Path, prefix: str):
    token = abs(hash(str(path.resolve())))
    name = f"_secat_{prefix}_{token}"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _ordered_api_names(history: Iterable[dict]) -> Tuple[str, ...]:
    names: List[str] = []
    seen = set()
    for item in history:
        if item.get("role") == "API":
            name = str(item.get("api_name", "")).strip()
            if name and name not in seen:
                seen.add(name)
                names.append(name)
    return tuple(names)


class APIBankRuntime:
    """Isolated API-Bank Level-1/Level-2 loader + official execution evaluator."""

    def __init__(self, vendor_root: Optional[os.PathLike] = None, *, state_mode: str = "published", level: int = 1):
        if vendor_root is None:
            configured = os.getenv("APIBANK_ROOT", "").strip()
            if configured:
                vendor_root = configured
            else:
                vendor_root = Path(__file__).resolve().parent / "api_bank_vendor"
        state_mode = str(state_mode or "published").strip().lower()
        if state_mode not in {"published", "replay"}:
            raise ValueError("state_mode must be published or replay")
        self.state_mode = state_mode
        try:
            self.level = int(level)
        except Exception as exc:
            raise ValueError("API-Bank level must be 1 or 2") from exc
        if self.level not in {1, 2}:
            raise ValueError("API-Bank level must be 1 or 2")
        self.root = discover_apibank_root(Path(vendor_root))
        if self.level == 2 and not self.level2_dir.is_dir():
            raise FileNotFoundError(f"API-Bank Level-2 data directory is missing: {self.level2_dir}")
        self._tool_manager_module = load_vendor_module(self.root, "tool_manager")
        self._parser_module = load_vendor_module(self.root, "api_call_extraction")
        if not hasattr(self._tool_manager_module, "ToolManager"):
            raise RuntimeError("API-Bank tool_manager.py has no ToolManager")
        if not hasattr(self._parser_module, "parse_api_call"):
            raise RuntimeError("API-Bank api_call_extraction.py has no parse_api_call")

    @property
    def level1_dir(self) -> Path:
        return self.root / "lv1-lv2-samples" / "level-1-given-desc"

    @property
    def level2_dir(self) -> Path:
        return self.root / "lv1-lv2-samples" / "level-2-toolsearcher"

    @property
    def level_dir(self) -> Path:
        return self.level1_dir if self.level == 1 else self.level2_dir

    @property
    def benchmark_name(self) -> str:
        return f"apibank_lv{self.level}"

    def _manager(self, api_names=None):
        cls = tool_manager_class(self.root, api_names, base_class=self._tool_manager_module.ToolManager)
        return cls("apis")

    def preflight(self, samples):
        """Validate selected API-Bank resources without making any LLM call.

        The official API-Bank checker implementations are not uniformly safe to
        invoke as reflexive preflight tests.  Some checkers assume task-specific
        response fields, normalize values asymmetrically, or depend on shapes
        that only exist for real predictions.  Therefore preflight verifies the
        *checker contract* (API importability + callable checker) but never calls
        ``check_api_call_correctness`` on synthetic/self data.  The unchanged
        official checker is still invoked for every scored prediction.

        Level 2 additionally validates every stored target structurally and
        executes one unscored ToolSearcher query to warm/verify the local
        retrieval stack before the helper script forces scored runs offline.
        """
        samples = list(samples)
        names = sorted({name for sample in samples for name in sample.api_names})
        manager = self._manager(names)
        checker_callable_tests = 0
        target_contract_tests = 0
        toolsearcher_samples = 0
        toolsearcher_runtime_tests = 0
        try:
            with _vendor_context(self.root, names), execution_guard():
                tools = {}
                for name in names:
                    tools[name] = manager.init_tool(name)
                if "SearchEngine" in names:
                    from nltk.tokenize import word_tokenize
                    try:
                        word_tokenize("API Bank preflight")
                    except LookupError as exc:
                        raise RuntimeError("API-Bank SearchEngine requires NLTK punkt_tab data; run python -m nltk.downloader punkt_tab") from exc

                if self.level == 2:
                    # Structural/contract validation only.  Do not invoke the
                    # official checker here: published API-Bank checkers are not
                    # guaranteed to be reflexive on stored ground truth.
                    for sample in samples:
                        name = str(sample.ground_truth.get("api_name") or "").strip()
                        if not name:
                            raise RuntimeError(
                                f"API-Bank target is missing api_name for {sample.task_id}")
                        if name not in tools:
                            try:
                                tools[name] = manager.init_tool(name)
                            except Exception as exc:
                                raise RuntimeError(
                                    f"API-Bank target API failed to initialize for {sample.task_id}/{name}: {exc}") from exc
                        api = tools[name]
                        checker = getattr(api, "check_api_call_correctness", None)
                        if not callable(checker):
                            raise RuntimeError(
                                f"API-Bank target checker is missing/not callable for {sample.task_id}/{name}")
                        checker_callable_tests += 1

                        if "result" not in sample.ground_truth:
                            raise RuntimeError(
                                f"API-Bank target is missing stored result for {sample.task_id}/{name}")
                        expected = sample.ground_truth.get("result")
                        if expected is not None and not isinstance(expected, dict):
                            raise RuntimeError(
                                f"API-Bank target result has unsupported type for {sample.task_id}/{name}: "
                                f"{type(expected).__name__}")

                        if name == "ToolSearcher":
                            if not isinstance(expected, dict) or "output" not in expected:
                                raise RuntimeError(
                                    f"API-Bank ToolSearcher ground truth has invalid result shape for {sample.task_id}")
                            output = expected.get("output")
                            if output is not None and not isinstance(output, (dict, list, str)):
                                raise RuntimeError(
                                    f"API-Bank ToolSearcher ground truth has unsupported output type for {sample.task_id}: "
                                    f"{type(output).__name__}")
                            toolsearcher_samples += 1
                        target_contract_tests += 1

                    # ToolSearcher's SentenceTransformer is loaded lazily in
                    # call(), not __init__(). Exercise it once here so the helper
                    # script can safely force scored runs offline afterwards.
                    if "ToolSearcher" in names:
                        try:
                            warm = manager.api_call("ToolSearcher", keywords="user token")
                        except Exception as exc:
                            raise RuntimeError(
                                "API-Bank ToolSearcher runtime warm-up failed; verify sentence-transformers "
                                f"and the local Hugging Face cache/network: {exc}") from exc
                        if not isinstance(warm, dict):
                            raise RuntimeError(
                                "API-Bank ToolSearcher warm-up returned a non-object result")
                        if warm.get("exception") not in (None, ""):
                            raise RuntimeError(
                                "API-Bank ToolSearcher warm-up returned an exception: "
                                + str(warm.get("exception")))
                        if "output" not in warm:
                            raise RuntimeError(
                                "API-Bank ToolSearcher warm-up result is missing 'output'")
                        output = warm.get("output")
                        if not isinstance(output, (dict, list)):
                            raise RuntimeError(
                                "API-Bank ToolSearcher warm-up returned invalid output type: "
                                + type(output).__name__)
                        items = [output] if isinstance(output, dict) else list(output)
                        if not items or not all(isinstance(item, dict) and str(item.get("name") or "").strip()
                                                for item in items):
                            raise RuntimeError(
                                "API-Bank ToolSearcher warm-up returned malformed tool descriptions")
                        toolsearcher_runtime_tests = 1
        finally:
            manager.close()
        return {
            "api_names": names,
            "samples": len(samples),
            "state_mode": self.state_mode,
            # Kept at zero for manifest/backward compatibility.  Semantic
            # checker self-tests are intentionally disabled in r35.
            "checker_self_tests": 0,
            "checker_callable_tests": checker_callable_tests,
            "target_contract_tests": target_contract_tests,
            "toolsearcher_samples": toolsearcher_samples,
            "toolsearcher_runtime_tests": toolsearcher_runtime_tests,
            "validated_targets": target_contract_tests,
            "live_api_names": [name for name in names if name in {"Translate", "Dictionary"}],
        }


    def _parse_prediction(self, text: str):
        parser = self._parser_module.parse_api_call
        with _vendor_context(self.root):
            # The ToolCoder snapshot exposes parse_api_call(text, action_mode),
            # while the original API-Bank/CAMEL copy exposes parse_api_call(text).
            # Detect the signature rather than catching TypeError from inside the
            # parser, which could otherwise hide a genuine parse failure.
            parameters = list(inspect.signature(parser).parameters.values())
            positional = [p for p in parameters
                          if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
            if len(positional) >= 2 and positional[1].default is inspect._empty:
                return parser(text, "code_as_action")
            return parser(text)

    @staticmethod
    def _toolsearcher_output_items(history: Iterable[dict]) -> Optional[List[dict]]:
        """Return the latest visible ToolSearcher result, normalized for display only.

        The ToolCoder/API-Bank LV2 corpus contains both list-valued and single-dict
        search outputs.  Upstream intends both forms to describe the tools visible
        after the search.  This normalization affects only the model-visible catalog;
        execution/checker values remain untouched.
        """
        latest = None
        for item in history:
            if item.get("role") != "API" or str(item.get("api_name")) != "ToolSearcher":
                continue
            result = item.get("result")
            output = result.get("output") if isinstance(result, dict) else None
            latest = output
        if latest is None:
            return None
        if isinstance(latest, dict):
            return [deepcopy(latest)]
        if isinstance(latest, list):
            return [deepcopy(x) for x in latest if isinstance(x, dict)]
        if isinstance(latest, str):
            try:
                parsed = json.loads(latest)
            except Exception:
                return []
            if isinstance(parsed, dict):
                return [parsed]
            if isinstance(parsed, list):
                return [deepcopy(x) for x in parsed if isinstance(x, dict)]
        return []

    def _visible_api_names(self, history: Iterable[dict]) -> Tuple[str, ...]:
        if self.level == 1:
            return ()
        items = self._toolsearcher_output_items(history)
        if items is None:
            return ("ToolSearcher",)
        names = []
        seen = set()
        for item in items:
            name = str(item.get("name") or "").strip()
            if name and name not in seen:
                seen.add(name)
                names.append(name)
        return tuple(names)

    def _load_samples(self, data_dir: Path, *, level: int) -> List[APIBankSample]:
        samples: List[APIBankSample] = []
        counter = 0
        for path in sorted(data_dir.glob("*.jsonl")):
            history: List[dict] = []
            with path.open("r", encoding="utf-8") as fh:
                for lineno, line in enumerate(fh, 1):
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid JSON in {path}:{lineno}: {exc}") from exc
                    if not isinstance(obj, dict):
                        raise ValueError(f"Expected object in {path}:{lineno}")
                    history.append(obj)

            api_names = _ordered_api_names(history)
            api_ordinal = 0
            for index, item in enumerate(history):
                if item.get("role") != "API":
                    continue
                api_ordinal += 1
                counter += 1
                prefix = tuple(deepcopy(history[:index]))
                visible = api_names if level == 1 else self._visible_api_names(prefix)
                samples.append(APIBankSample(
                    task_id=f"apibank_lv{level}_{counter:03d}",
                    source_file=path.name,
                    source_api_index=api_ordinal,
                    chat_history=prefix,
                    api_names=api_names,
                    ground_truth=deepcopy(item),
                    benchmark_level=level,
                    visible_api_names=tuple(visible),
                ))
        if not samples:
            raise RuntimeError(f"No Level-{level} API-call samples found in {data_dir}")
        return samples

    def load_level1_samples(self) -> List[APIBankSample]:
        return self._load_samples(self.level1_dir, level=1)

    def load_level2_samples(self) -> List[APIBankSample]:
        return self._load_samples(self.level2_dir, level=2)

    def load_samples(self) -> List[APIBankSample]:
        return self.load_level1_samples() if self.level == 1 else self.load_level2_samples()

    @staticmethod
    def _history_for_agent(history: Iterable[dict]) -> List[dict]:
        safe: List[dict] = []
        for item in history:
            role = item.get("role")
            if role == "API":
                # Prior API results are legitimate conversation evidence.  The
                # target API turn is not part of chat_history and therefore can
                # never leak here.
                safe.append({
                    "role": "API",
                    "api_name": item.get("api_name"),
                    "param_dict": deepcopy(item.get("param_dict") or {}),
                    "result": deepcopy(item.get("result")),
                })
            else:
                safe.append({"role": role, "text": item.get("text", "")})
        return safe

    def _catalog_for_names(self, names: Iterable[str]) -> List[dict]:
        names = tuple(str(x) for x in names if str(x).strip())
        manager = self._manager(names)
        catalog: List[dict] = []
        try:
            for name in names:
                with _vendor_context(self.root, names):
                    description = str(manager.get_api_description(name))
                    metadata = {}
                    getter = getattr(manager, "get_api_by_name", None)
                    if callable(getter):
                        try:
                            value = getter(name)
                            if isinstance(value, dict):
                                metadata = deepcopy({k: v for k, v in value.items() if k in {"name", "description", "input_parameters", "output_parameters"}})
                        except Exception:
                            metadata = {}
                parsed = None
                try:
                    value = json.loads(description)
                    if isinstance(value, dict):
                        parsed = value
                except Exception:
                    parsed = None
                catalog.append({
                    "name": str(name),
                    "description": description,
                    "description_json": parsed,
                    "metadata": metadata,
                })
        finally:
            manager.close()
        return catalog

    @staticmethod
    def _catalog_from_search_items(items: Iterable[dict]) -> List[dict]:
        catalog = []
        for raw in items or []:
            if not isinstance(raw, dict):
                continue
            metadata = deepcopy({k: v for k, v in raw.items()
                                 if k in {"name", "description", "input_parameters", "output_parameters"}})
            name = str(raw.get("name") or "").strip()
            if not name:
                continue
            catalog.append({
                "name": name,
                "description": json.dumps(raw, ensure_ascii=False),
                "description_json": deepcopy(raw),
                "metadata": metadata,
            })
        return catalog

    def api_catalog(self, sample: APIBankSample) -> List[dict]:
        """Return exactly the API metadata visible under the benchmark protocol."""
        if int(getattr(sample, "benchmark_level", self.level) or self.level) == 2:
            items = self._toolsearcher_output_items(sample.chat_history)
            if items is None:
                return self._catalog_for_names(("ToolSearcher",))
            return self._catalog_from_search_items(items)
        return self._catalog_for_names(sample.api_names)

    def api_descriptions(self, sample: APIBankSample) -> str:
        return "\n".join(entry["description"] for entry in self.api_catalog(sample))

    def agent_view(self, sample: APIBankSample) -> dict:
        """Return the only benchmark information allowed across agent boundary."""
        return {
            "task_id": sample.task_id,
            "benchmark": f"apibank_lv{int(getattr(sample, 'benchmark_level', self.level) or self.level)}",
            "benchmark_level": int(getattr(sample, "benchmark_level", self.level) or self.level),
            "api_descriptions": self.api_descriptions(sample),
            "chat_history": self._history_for_agent(sample.chat_history),
        }

    @staticmethod
    def redacted_sample(sample: APIBankSample) -> APIBankSample:
        """Return execution context with target answer/future-tool names removed."""
        safe = sample.with_ground_truth({})
        if int(getattr(sample, "benchmark_level", 1) or 1) == 2:
            safe = replace(safe, api_names=tuple(sample.visible_api_names))
        return safe

    def benchmark_fingerprint(self) -> str:
        """Hash executable benchmark inputs, not just the dialogue JSONL files."""
        files = [self.root / "tool_manager.py", self.root / "api_call_extraction.py"]
        template = self.root / "template.py"
        if template.is_file():
            files.append(template)
        files.extend(sorted((self.root / "apis").rglob("*.py")))
        init_db = self.root / "init_database"
        if init_db.is_dir():
            files.extend(sorted(init_db.rglob("*.json")))
        files.extend(sorted(self.level_dir.rglob("*.jsonl")))
        digest = hashlib.sha256()
        for path in sorted({p.resolve() for p in files}, key=lambda x: str(x.relative_to(self.root))):
            rel = str(path.relative_to(self.root)).replace(os.sep, "/")
            digest.update(rel.encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
        return digest.hexdigest()

    def _replay(self, manager, sample: APIBankSample) -> int:
        replayed = 0
        for item in sample.chat_history:
            if item.get("role") != "API":
                continue
            name = item.get("api_name")
            params = deepcopy(item.get("param_dict") or {})
            with _vendor_context(self.root):
                manager.api_call(name, **params)
            replayed += 1
        return replayed

    def _manager_for_sample(self, sample: APIBankSample, extra_names=()):
        manager = self._manager(tuple(sample.api_names) + tuple(extra_names))
        replayed = 0
        if self.state_mode == "replay":
            replayed = self._replay(manager, sample)
        return manager, replayed

    def execute_for_agent(self, sample: APIBankSample, api_name: str, params: dict):
        """Execute one agent action using the configured state protocol."""
        manager, _ = self._manager_for_sample(sample, (api_name,))
        with _vendor_context(self.root):
            return manager.api_call(api_name, **deepcopy(params or {}))

    def evaluate_executed(
        self, sample: APIBankSample, *, api_name: str, params: Optional[dict],
        result: Any, replayed_calls: int = 0, _manager=None
    ) -> EvaluationResult:
        """Score an already-executed action without executing the target again.

        ``_manager`` is private on purpose: the published CodeAct/API-Bank
        evaluator checks the result on the same ToolManager instance that
        executed the API call.  Externally executed ToolCoder calls have no
        in-process manager, so they are checked with a fresh manager.
        """
        expected_name = str(sample.ground_truth.get("api_name", ""))
        if str(api_name) != expected_name:
            return EvaluationResult(
                False, predicted_api=str(api_name), predicted_params=deepcopy(params or {}),
                execution_result=deepcopy(result), replayed_calls=int(replayed_calls),
                error_type="api_name_mismatch",
                error=f"predicted {api_name!r}; expected {expected_name!r}")
        expected_result = sample.ground_truth.get("result")
        if isinstance(expected_result, dict) and (
            not isinstance(result, dict) or not set(expected_result).issubset(result)
        ):
            return EvaluationResult(False, predicted_api=str(api_name),
                predicted_params=deepcopy(params or {}), execution_result=deepcopy(result),
                replayed_calls=int(replayed_calls), error_type="malformed_result",
                error="Printed API result does not satisfy the expected response envelope")
        try:
            manager = _manager if _manager is not None else self._manager((api_name,))
            with _vendor_context(self.root), execution_guard():
                api = manager.init_tool(api_name)
                correct = api.check_api_call_correctness(
                    deepcopy(result), deepcopy(sample.ground_truth.get("result")))
        except APIBankInfrastructureError as exc:
            return EvaluationResult(False, predicted_api=str(api_name), predicted_params=deepcopy(params or {}),
                execution_result=deepcopy(result), replayed_calls=int(replayed_calls),
                error_type=exc.error_type, error=str(exc))
        except KeyError as exc:
            # This is exactly how the published API-Bank evaluator treats a
            # checker KeyError: a normal incorrect prediction, not harness loss.
            return EvaluationResult(
                False, predicted_api=str(api_name), predicted_params=deepcopy(params or {}),
                execution_result=deepcopy(result), replayed_calls=int(replayed_calls),
                error_type="official_checker_mismatch", error=f"KeyError: {exc}")
        except Exception as exc:
            # A bad printed input can fail inside .strip()/indexing in an official
            # checker. Only classify proven external-result shape errors this way;
            # same-shape checker defects remain infrastructure failures.
            def incompatible(actual, expected):
                if isinstance(expected, dict):
                    return not isinstance(actual, dict) or any(
                        k not in actual or incompatible(actual[k], v) for k, v in expected.items())
                if isinstance(expected, (list, tuple)):
                    return not isinstance(actual, (list, tuple))
                if expected is None:
                    return False
                if isinstance(expected, (int, float)) and not isinstance(expected, bool):
                    return not isinstance(actual, (int, float))
                return not isinstance(actual, type(expected))
            malformed = (_manager is None and isinstance(exc, (TypeError, AttributeError, IndexError))
                         and incompatible(result, expected_result))
            return EvaluationResult(
                False, predicted_api=str(api_name), predicted_params=deepcopy(params or {}),
                execution_result=deepcopy(result), replayed_calls=int(replayed_calls),
                error_type="malformed_result" if malformed else "checker_error", error=str(exc))
        finally:
            if "manager" in locals():
                manager.close()
        return EvaluationResult(
            bool(correct), predicted_api=str(api_name), predicted_params=deepcopy(params or {}),
            execution_result=deepcopy(result), replayed_calls=int(replayed_calls),
            error_type=None if bool(correct) else "official_checker_mismatch")

    def evaluate(self, sample: APIBankSample, prediction_text: str) -> EvaluationResult:
        try:
            api_name, param_dict = self._parse_prediction(str(prediction_text or ""))
            param_dict = dict(param_dict or {})
        except Exception as exc:
            return EvaluationResult(False, error_type="parse_error", error=str(exc))

        return self.evaluate_action(sample, api_name, param_dict)

    def evaluate_action(self, sample: APIBankSample, api_name: str, params: dict) -> EvaluationResult:
        """Score structured actions without a lossy text/parser round trip."""
        if not isinstance(params, dict):
            return EvaluationResult(False, error_type="parse_error", error="Action params must be an object")
        param_dict = deepcopy(params)
        expected_name = str(sample.ground_truth.get("api_name", ""))
        if str(api_name) != expected_name:
            return EvaluationResult(
                False, predicted_api=str(api_name), predicted_params=deepcopy(param_dict),
                error_type="api_name_mismatch",
                error=f"predicted {api_name!r}; expected {expected_name!r}")

        replayed = 0
        try:
            manager, replayed = self._manager_for_sample(sample, (api_name,))
        except APIBankInfrastructureError as exc:
            return EvaluationResult(False, predicted_api=str(api_name), predicted_params=deepcopy(param_dict),
                replayed_calls=replayed, error_type=exc.error_type, error=str(exc))
        except Exception as exc:
            return EvaluationResult(
                False, predicted_api=str(api_name), predicted_params=deepcopy(param_dict),
                replayed_calls=replayed, error_type="replay_error", error=str(exc))

        try:
            with _vendor_context(self.root):
                result = manager.api_call(api_name, **deepcopy(param_dict))
        except APIBankInfrastructureError as exc:
            return EvaluationResult(False, predicted_api=str(api_name), predicted_params=deepcopy(param_dict),
                replayed_calls=replayed, error_type=exc.error_type, error=str(exc))
        except Exception as exc:
            return EvaluationResult(
                False, predicted_api=str(api_name), predicted_params=deepcopy(param_dict),
                replayed_calls=replayed, error_type="execution_error", error=str(exc))

        return self.evaluate_executed(
            sample, api_name=str(api_name), params=param_dict, result=result,
            replayed_calls=replayed, _manager=manager)

