"""Provider-neutral propagation of terminal external-runtime signals.

Execution backends stay provider agnostic. Provider-specific transports can emit a
small, explicit terminal marker; this module turns it into the corresponding trusted
host exception so orchestration can stop without sending more external calls.
"""
from __future__ import annotations

import os


def raise_if_fatal_runtime_signal(output: str) -> None:
    text = str(output or "")
    if "QUOTA_EXCEEDED" not in text:
        return
    runtime = str(os.environ.get("SECAT_BENCHMARK") or "").casefold()
    # Isolated runs use the runtime API label, while compatibility runs may use a
    # registered benchmark name. Resolve through the registry rather than baking a
    # provider branch into the execution backend.
    try:
        import benchmarks as B
        spec = B.get_benchmark(runtime)
        runtime = str(spec.get("runtime_api") or spec.get("name") or runtime).casefold()
    except Exception:
        pass
    if runtime == "spotify":
        from utils.spotify_runtime import SpotifyRateLimitError
        raise SpotifyRateLimitError(
            "Spotify development-mode quota is exhausted (QUOTA_EXCEEDED); "
            "aborting the batch before additional agent/API calls",
            reason="QUOTA_EXCEEDED", quota_exceeded=True)
