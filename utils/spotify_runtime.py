"""Spotify benchmark runtime, safety guard, and 2026 compatibility layer.

The RestBench-Spotify corpus is stateful and contains write operations.  This
module gives CodeAct, ToolCoder, and OCA the same transport semantics:

* ``legacy`` profile: execute the original RestBench endpoint paths unchanged.
* ``dev2026`` profile: rewrite only documented, semantics-preserving endpoint
  migrations.  Endpoints with no faithful replacement fail explicitly.
* write requests are blocked unless ``SECAT_ALLOW_SPOTIFY_WRITES=YES``.

Credentials are captured by trusted runtime code. Access tokens are refreshed once on
HTTP 401 when refresh credentials are available.  When the project already has an
``.env`` file, refreshed OAuth material is atomically persisted there so rotated refresh
tokens survive separate CodeAct/ToolCoder/OCA processes and ToolCoder child executions.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from copy import deepcopy
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit

import requests as _requests

BASE_URL = "https://api.spotify.com/v1"
WRITE_METHODS = {"POST", "PUT", "DELETE", "PATCH"}
SUPPORTED_PROFILES = {"legacy", "dev2026"}

# Capture OAuth material in trusted infrastructure before isolated execution removes
# SPOTIFY_ACCESS_TOKEN from generated-code environments.  These values are never
# added to prompts or observations.
_RUNTIME_ACCESS_TOKEN = os.environ.get("SPOTIFY_ACCESS_TOKEN", "")
_REFRESH_TOKEN = os.environ.get("SPOTIFY_REFRESH_TOKEN", "")
_CLIENT_ID = os.environ.get("SPOTIFY_CLIENT_ID", "")
_CLIENT_SECRET = os.environ.get("SPOTIFY_CLIENT_SECRET", "")

# Broad enough for the original corpus.  Exact authorization still depends on
# the app/account and Spotify's current policy.
REQUIRED_SCOPES = (
    "user-read-private", "user-read-email", "playlist-read-private",
    "playlist-read-collaborative",
    "playlist-modify-private", "playlist-modify-public", "user-library-read",
    "user-library-modify", "user-follow-read", "user-follow-modify",
    "user-read-playback-state", "user-modify-playback-state",
    "user-read-currently-playing", "user-read-recently-played", "user-top-read",
)


class SpotifyBenchmarkError(RuntimeError):
    pass


class SpotifyWriteBlocked(SpotifyBenchmarkError):
    pass


class SpotifyEndpointUnsupported(SpotifyBenchmarkError):
    pass


class SpotifyRateLimitError(SpotifyBenchmarkError):
    """A trusted fixture request remained rate-limited after transport policy."""

    def __init__(self, message: str, *, retry_after: float | None = None,
                 reason: str = "", quota_exceeded: bool = False):
        super().__init__(message)
        self.retry_after = retry_after
        self.reason = reason
        self.quota_exceeded = bool(quota_exceeded)


_RATE_LOCK = threading.Lock()
_LAST_SPOTIFY_REQUEST_AT = 0.0
_FIXTURE_CACHE_LOCK = threading.Lock()
_FIXTURE_SEARCH_CACHE: dict[str, list[dict[str, Any]]] = {}
_FIXTURE_CACHE_LOADED_PATHS: set[str] = set()


def _bounded_env_float(name: str, default: float, *, minimum: float = 0.0,
                       maximum: float = 600.0) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return min(maximum, max(minimum, value))


def _bounded_env_int(name: str, default: int, *, minimum: int = 0,
                     maximum: int = 10) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return min(maximum, max(minimum, value))


def _wait_for_spotify_request_slot() -> float:
    """Pace physical Web API calls within this process using monotonic time."""
    global _LAST_SPOTIFY_REQUEST_AT
    interval = _bounded_env_float(
        "SPOTIFY_MIN_REQUEST_INTERVAL_SEC", 1.0, maximum=30.0)
    if interval <= 0:
        return 0.0
    waited = 0.0
    with _RATE_LOCK:
        now = time.monotonic()
        delay = interval - (now - _LAST_SPOTIFY_REQUEST_AT)
        if _LAST_SPOTIFY_REQUEST_AT > 0 and delay > 0:
            time.sleep(delay)
            waited = delay
        _LAST_SPOTIFY_REQUEST_AT = time.monotonic()
    return waited


def _rate_limit_details(response: Any) -> tuple[float | None, str, bool]:
    headers = getattr(response, "headers", {}) or {}
    raw_retry = headers.get("Retry-After") or headers.get("retry-after")
    retry_after = None
    try:
        if raw_retry not in (None, ""):
            retry_after = max(0.0, float(raw_retry))
    except (TypeError, ValueError):
        retry_after = None
    reason = ""
    try:
        payload = response.json() or {}
        error = payload.get("error") if isinstance(payload, dict) else {}
        if isinstance(error, dict):
            reason = str(error.get("reason") or error.get("message") or "")
    except Exception:
        reason = ""
    quota_exceeded = reason.strip().upper() == "QUOTA_EXCEEDED"
    return retry_after, reason, quota_exceeded


def _perform_spotify_request(method: str, url: str, **kwargs) -> tuple[Any, int]:
    """Execute one physical Spotify request with pacing and bounded 429 retry.

    Normal responses, including an exhausted final 429, retain requests-compatible
    behavior. Trusted fixture wrappers separately turn a final 429 into a clear
    infrastructure exception so agent code is not given different semantics.
    """
    retries = _bounded_env_int("SPOTIFY_RATE_LIMIT_RETRIES", 4)
    fallback = _bounded_env_float(
        "SPOTIFY_RATE_LIMIT_BACKOFF_SEC", 1.0, maximum=60.0)
    max_wait = _bounded_env_float(
        "SPOTIFY_MAX_RETRY_AFTER_SEC", 300.0, maximum=3600.0)
    attempts = 0
    total_wait = 0.0
    response = None
    for retry_index in range(retries + 1):
        total_wait += _wait_for_spotify_request_slot()
        response = _requests.request(method, url, **kwargs)
        attempts += 1
        if getattr(response, "status_code", None) != 429:
            break
        retry_after, _reason, quota_exceeded = _rate_limit_details(response)
        if quota_exceeded or retry_index >= retries:
            break
        if retry_after is not None:
            # A very large Retry-After is a run-level stop condition rather than
            # permission for one hidden request to block indefinitely.
            if retry_after > max_wait:
                break
            delay = retry_after
        else:
            delay = min(max_wait, fallback * (2 ** retry_index))
        if delay > 0:
            time.sleep(delay)
            total_wait += delay
    assert response is not None
    retry_after, reason, quota_exceeded = _rate_limit_details(response)
    try:
        response.secat_rate_limit = {
            "attempts": attempts,
            "retries": max(0, attempts - 1),
            "wait_seconds": round(total_wait, 6),
            "retry_after": retry_after,
            "reason": reason,
            "quota_exceeded": quota_exceeded,
        }
    except Exception:
        pass
    return response, attempts


def _rate_limit_error(response: Any) -> SpotifyRateLimitError:
    retry_after, reason, quota_exceeded = _rate_limit_details(response)
    stats = getattr(response, "secat_rate_limit", {}) or {}
    attempts = int(stats.get("attempts") or 1)
    if quota_exceeded:
        message = (
            "Spotify development-mode quota is exhausted (QUOTA_EXCEEDED); "
            "stopping fixture setup without repeated task crashes")
    else:
        suffix = f"; Retry-After={retry_after:g}s" if retry_after is not None else ""
        message = (
            f"Spotify remained rate-limited after {attempts} request attempt(s){suffix}; "
            "stopping fixture setup")
    return SpotifyRateLimitError(
        message, retry_after=retry_after, reason=reason,
        quota_exceeded=quota_exceeded)


def profile() -> str:
    value = os.environ.get("SPOTIFY_API_PROFILE", "legacy").strip().lower()
    if value not in SUPPORTED_PROFILES:
        raise SpotifyBenchmarkError(
            f"invalid SPOTIFY_API_PROFILE={value!r}; choose legacy or dev2026")
    return value


def writes_allowed() -> bool:
    return os.environ.get("SECAT_ALLOW_SPOTIFY_WRITES", "").strip().upper() == "YES"


def validate_task_selection(tasks: list[dict], *, selected_profile: str | None = None,
                            allow_writes: bool | None = None) -> dict:
    """Validate only run-level Spotify settings before execution.

    The pre-execution harness deliberately does *not* inspect benchmark
    ``solution``/``solution_steps`` or task-number allow/deny lists.  Doing so
    would let oracle metadata influence which tasks are admitted to the run.
    Write safety and dev-profile endpoint support are enforced dynamically by
    the trusted transport from the request that the agent actually makes.
    """
    selected_profile = (selected_profile or profile()).strip().lower()
    if selected_profile not in SUPPORTED_PROFILES:
        raise SpotifyBenchmarkError(
            f"invalid Spotify profile {selected_profile!r}; choose legacy or dev2026")
    allow_writes = writes_allowed() if allow_writes is None else bool(allow_writes)
    return {
        "tasks": len(tasks),
        "profile": selected_profile,
        "writes_enabled": bool(allow_writes),
        "preexecution_oracle_inspection": False,
        "write_tasks": None,
        "read_only_tasks": None,
        "unsupported": [],
    }


def _spotify_path(url: str) -> tuple[str, dict[str, list[str]]]:
    parts = urlsplit(url)
    path = parts.path
    if path.startswith("/v1/"):
        path = path[3:]
    elif path == "/v1":
        path = "/"
    return path, parse_qs(parts.query, keep_blank_values=True)


def _absolute(path: str, original_url: str = BASE_URL) -> str:
    if path.startswith("http://") or path.startswith("https://"):
        return path
    return BASE_URL.rstrip("/") + "/" + path.lstrip("/")


def _ids_from(params: dict | None, body: Any, query: dict[str, list[str]]) -> list[str]:
    values: Any = None
    for source in (params or {}, body if isinstance(body, dict) else {}, query):
        if "ids" in source:
            values = source["ids"]
            break
    if isinstance(values, list):
        if len(values) == 1 and isinstance(values[0], str) and "," in values[0]:
            values = values[0].split(",")
        return [str(x) for x in values]
    if isinstance(values, str):
        return [x for x in values.split(",") if x]
    return []


def _library_uris(kind: str, ids: list[str]) -> list[str]:
    prefix = {"tracks": "track", "albums": "album", "following": "artist",
              "playlists": "playlist"}[kind]
    return [x if x.startswith("spotify:") else f"spotify:{prefix}:{x}" for x in ids]


@dataclass
class AdaptedRequest:
    method: str
    url: str
    kwargs: dict[str, Any]
    original_method: str
    original_path: str
    adapted: bool = False
    note: str | None = None
    search_target_limit: int | None = None
    search_start_offset: int = 0
    paging_target_limit: int | None = None
    paging_start_offset: int = 0
    paging_page_cap: int | None = None
    library_uris: list[str] | None = None


def adapt_request(method: str, url: str, kwargs: dict[str, Any] | None = None,
                  selected_profile: str | None = None) -> AdaptedRequest:
    """Return a transport request under the selected compatibility profile."""
    method = method.upper()
    kwargs = dict(kwargs or {})
    path, query = _spotify_path(url)
    result = AdaptedRequest(method, url, kwargs, method, path)
    if (selected_profile or profile()) == "legacy":
        return result

    body = kwargs.get("json")
    if body is None and isinstance(kwargs.get("data"), dict):
        body = dict(kwargs["data"])

    if method == "GET" and (
            re.fullmatch(r"/artists/[^/]+/top-tracks", path)
            or re.fullmatch(r"/artists/[^/]+/related-artists", path)):
        raise SpotifyEndpointUnsupported(
            f"{method} {path} is outside the restricted dev2026 endpoint set "
            "and has no faithful benchmark-preserving replacement")

    if method == "GET" and re.fullmatch(r"/artists/[^/]+/albums", path):
        # Development Mode reduced Get Artist's Albums to default 5 / max 10.
        # Preserve historical callers (whose bundled OAS advertises a larger
        # default/max) by transparently paginating.  The trace still records one
        # canonical benchmark call, while the response exposes the requested
        # legacy-sized item list.
        params = dict(result.kwargs.get("params") or {})
        for key, values in query.items():
            if key not in params:
                params[key] = values[0] if len(values) == 1 else list(values)
        requested = params.get("limit", 20)
        try:
            target = int(requested)
            start_offset = int(params.get("offset", 0))
        except (TypeError, ValueError) as exc:
            raise SpotifyBenchmarkError(
                f"invalid artist-albums pagination parameters: limit={requested!r}, "
                f"offset={params.get('offset')!r}") from exc
        if target < 1 or target > 50 or start_offset < 0:
            raise SpotifyBenchmarkError(
                f"artist-albums legacy range is limit 1..50 and offset >=0; got "
                f"limit={target}, offset={start_offset}")
        parts = urlsplit(result.url)
        result.url = urlunsplit((parts.scheme, parts.netloc, parts.path, "", parts.fragment))
        params["limit"] = min(10, target)
        params["offset"] = start_offset
        result.kwargs["params"] = params
        result.paging_target_limit = target
        result.paging_start_offset = start_offset
        result.paging_page_cap = 10
        if target > 10 or "limit" not in (kwargs.get("params") or {}):
            result.adapted = True
            result.note = (f"Artist albums limit {target} preserved with offset pagination "
                           "in pages of at most 10")

    if method == "GET" and path == "/search":
        # Development Mode reduced Search from legacy default/max 20/50 to 5/10.
        # Preserve the caller's historical request through transparent offset
        # pagination, emitted as one canonical benchmark call in the shared trace.
        params = dict(result.kwargs.get("params") or {})
        for key, values in query.items():
            if key not in params:
                params[key] = values[0] if len(values) == 1 else list(values)
        requested = params.get("limit", 20)
        try:
            target = int(requested)
            start_offset = int(params.get("offset", 0))
        except (TypeError, ValueError) as exc:
            raise SpotifyBenchmarkError(
                f"invalid Search pagination parameters: limit={requested!r}, "
                f"offset={params.get('offset')!r}") from exc
        if target < 1 or target > 50 or start_offset < 0:
            raise SpotifyBenchmarkError(
                f"Search legacy range is limit 1..50 and offset >=0; got "
                f"limit={target}, offset={start_offset}")
        parts = urlsplit(result.url)
        result.url = urlunsplit((parts.scheme, parts.netloc, parts.path, "", parts.fragment))
        params["limit"] = min(10, target)
        params["offset"] = start_offset
        result.kwargs["params"] = params
        result.search_target_limit = target
        result.search_start_offset = start_offset
        if target > 10:
            result.adapted = True
            result.note = (f"Search limit {target} preserved with offset pagination "
                           "in pages of at most 10")

    # Spotify playback always expects ``uris`` as a JSON array. A selected
    # singleton is still a collection-valued request field; normalize the scalar
    # shape instead of letting a semantically valid play action fail with HTTP 400.
    if method == "PUT" and path == "/me/player/play" and isinstance(body, dict):
        if "uris" in body and not isinstance(body.get("uris"), (list, tuple)):
            body = dict(body)
            body["uris"] = [body.get("uris")]
            result.kwargs["json"] = body
            result.kwargs.pop("data", None)
            result.adapted = True
            result.note = "normalized singleton playback URI to JSON array"

    m = re.fullmatch(r"/users/[^/]+/playlists", path)
    if method == "POST" and m:
        result.url = _absolute("/me/playlists")
        result.adapted, result.note = True, "POST /users/{id}/playlists -> POST /me/playlists"
        return result

    # Playlist item naming migration.
    if re.fullmatch(r"/playlists/[^/]+/tracks", path):
        new_path = path[:-len("/tracks")] + "/items"
        result.url = _absolute(new_path)
        if isinstance(body, dict) and "tracks" in body and "items" not in body:
            body = dict(body); body["items"] = body.pop("tracks")
            result.kwargs["json"] = body; result.kwargs.pop("data", None)
        # The historical endpoint accepted `uris` in the URL query. The new add
        # endpoint accepts the same values in JSON; convert rather than dropping
        # them when the supplied ToolCoder-style code embeds the query in the URL.
        if method == "POST" and query.get("uris") and not isinstance(body, dict):
            uris = []
            for value in query.get("uris", []):
                uris.extend(x for x in value.split(",") if x)
            result.kwargs["json"] = {"uris": uris}
        else:
            preserved = {k: v for k, v in query.items() if not (method == "POST" and k == "uris")}
            if preserved:
                result.url += "?" + urlencode(preserved, doseq=True)
        result.adapted, result.note = True, "playlist /tracks -> /items"
        return result

    # Save/remove/follow migrations to the unified library endpoint.
    library_kind = None
    if path in {"/me/tracks", "/me/albums", "/me/following"} and method in {"PUT", "DELETE"}:
        library_kind = path.rsplit("/", 1)[-1]
    if method == "DELETE" and re.fullmatch(r"/playlists/[^/]+/followers", path):
        library_kind = "playlists"
        ids = [path.split("/")[2]]
    else:
        ids = _ids_from(result.kwargs.get("params"), body, query)
    if library_kind:
        if not ids:
            raise SpotifyEndpointUnsupported(
                f"cannot adapt {method} {path}: no resource ids were supplied")
        uris = _library_uris(library_kind, ids)
        result.url = _absolute("/me/library")
        result.kwargs.pop("json", None); result.kwargs.pop("data", None)
        # Current Development Mode uses a required comma-separated `uris`
        # request parameter, with at most 40 URI values per request. Preserve
        # historical callers by transparently chunking larger sets in the
        # trusted transport while retaining one canonical benchmark trace.
        result.kwargs["params"] = {"uris": ",".join(uris[:40])}
        result.library_uris = uris
        result.adapted, result.note = True, f"{path} -> /me/library"
        if len(uris) > 40:
            result.note += f"; {len(uris)} URIs chunked in groups of 40"
        return result

    return result


def _active_access_token(fallback: str | None = None) -> str:
    global _RUNTIME_ACCESS_TOKEN
    if _RUNTIME_ACCESS_TOKEN:
        return _RUNTIME_ACCESS_TOKEN
    if fallback:
        _RUNTIME_ACCESS_TOKEN = str(fallback)
        return _RUNTIME_ACCESS_TOKEN
    value = os.environ.get("SPOTIFY_ACCESS_TOKEN", "")
    if value:
        _RUNTIME_ACCESS_TOKEN = value
    return _RUNTIME_ACCESS_TOKEN


def _oauth_env_path() -> Path:
    root = os.environ.get("SECAT_PROJECT_ROOT", "").strip()
    if root:
        return Path(root).expanduser().resolve() / ".env"
    return Path(__file__).resolve().parents[1] / ".env"


def _persist_oauth_tokens(access_token: str, refresh_token: str = "") -> bool:
    """Atomically persist refreshed Spotify OAuth material to an existing .env.

    This is trusted infrastructure only.  It never creates a new credential file,
    never writes tokens to logs, and can be disabled with
    ``SECAT_SPOTIFY_PERSIST_OAUTH=NO``.
    """
    if os.environ.get("SECAT_SPOTIFY_PERSIST_OAUTH", "YES").strip().upper() == "NO":
        return False
    path = _oauth_env_path()
    if not path.exists() or not path.is_file():
        return False
    try:
        original = path.read_text(encoding="utf-8").splitlines()
        out = []
        saw_access = False
        saw_refresh = False
        for line in original:
            if line.startswith("SPOTIFY_ACCESS_TOKEN="):
                out.append("SPOTIFY_ACCESS_TOKEN=" + access_token)
                saw_access = True
            elif refresh_token and line.startswith("SPOTIFY_REFRESH_TOKEN="):
                out.append("SPOTIFY_REFRESH_TOKEN=" + refresh_token)
                saw_refresh = True
            else:
                out.append(line)
        if not saw_access:
            out.append("SPOTIFY_ACCESS_TOKEN=" + access_token)
        if refresh_token and not saw_refresh:
            out.append("SPOTIFY_REFRESH_TOKEN=" + refresh_token)
        payload = "\n".join(out) + "\n"
        tmp = path.with_name(path.name + ".spotify-tmp")
        mode = path.stat().st_mode
        tmp.write_text(payload, encoding="utf-8")
        os.chmod(tmp, mode)
        os.replace(tmp, path)
        return True
    except Exception:
        try:
            tmp = path.with_name(path.name + ".spotify-tmp")
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        return False


def _refresh_access_token() -> str | None:
    """Refresh Spotify OAuth once in trusted infrastructure.

    This deliberately bypasses the Spotify API compatibility proxy and talks only
    to Spotify's OAuth token endpoint.  Refreshed credentials are persisted only to
    an already-existing project .env file, atomically and outside agent visibility.
    """
    global _RUNTIME_ACCESS_TOKEN, _REFRESH_TOKEN
    refresh = _REFRESH_TOKEN or os.environ.get("SPOTIFY_REFRESH_TOKEN", "")
    client_id = _CLIENT_ID or os.environ.get("SPOTIFY_CLIENT_ID", "")
    client_secret = _CLIENT_SECRET or os.environ.get("SPOTIFY_CLIENT_SECRET", "")
    if not (refresh and client_id and client_secret):
        return None
    try:
        response = _requests.post(
            "https://accounts.spotify.com/api/token",
            auth=(client_id, client_secret),
            data={"grant_type": "refresh_token", "refresh_token": refresh},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30,
        )
        if response.status_code != 200:
            return None
        payload = response.json() or {}
        token = str(payload.get("access_token") or "")
        if not token:
            return None
        _RUNTIME_ACCESS_TOKEN = token
        os.environ["SPOTIFY_ACCESS_TOKEN"] = token
        rotated = str(payload.get("refresh_token") or "")
        if rotated:
            _REFRESH_TOKEN = rotated
            os.environ["SPOTIFY_REFRESH_TOKEN"] = rotated
        _persist_oauth_tokens(token, rotated)
        return token
    except Exception:
        return None


def _prepare(method: str, url: str, kwargs: dict[str, Any], *, access_token: str | None = None) -> AdaptedRequest:
    if not str(url).startswith(BASE_URL):
        return AdaptedRequest(method.upper(), url, kwargs, method.upper(), url)
    if method.upper() in WRITE_METHODS and not writes_allowed():
        path, _ = _spotify_path(url)
        raise SpotifyWriteBlocked(
            f"blocked Spotify write {method.upper()} {path}; rerun with "
            "--allow-spotify-writes on a disposable benchmark account")
    token = _active_access_token(access_token)
    headers = dict(kwargs.get("headers") or {})
    if token and "Authorization" not in headers:
        headers["Authorization"] = f"Bearer {token}"
    kwargs = dict(kwargs); kwargs["headers"] = headers
    return adapt_request(method, url, kwargs)



def _rename_playlist_entries(paging: Any) -> Any:
    """Return a copy of a playlist-items paging object in legacy field shape.

    Spotify's restricted 2026 response schema renamed each playlist entry's
    ``item`` field to the historical ``track`` field.  RestBench and the
    supplied ToolCoder prompts were authored against the historical schema.
    """
    if not isinstance(paging, dict):
        return paging
    out = dict(paging)
    entries = out.get("items")
    if isinstance(entries, list):
        normalized = []
        for entry in entries:
            if isinstance(entry, dict):
                entry = dict(entry)
                if "item" in entry and "track" not in entry:
                    entry["track"] = entry.pop("item")
            normalized.append(entry)
        out["items"] = normalized
    return out


def _normalize_playlist_object(value: Any) -> Any:
    """Convert a dev2026 playlist object to the legacy RestBench field names."""
    if not isinstance(value, dict):
        return value
    out = dict(value)
    candidate = out.get("items")
    # A playlist object's renamed ``items`` member is itself a paging object.
    # Do not confuse it with an outer paging object's list-valued ``items``.
    if isinstance(candidate, dict) and isinstance(candidate.get("items"), list):
        if "tracks" not in out:
            out["tracks"] = _rename_playlist_entries(candidate)
        out.pop("items", None)
    elif isinstance(out.get("tracks"), dict):
        out["tracks"] = _rename_playlist_entries(out["tracks"])
    return out


def normalize_response_payload(payload: Any, original_path: str) -> tuple[Any, str | None]:
    """Restore legacy playlist response names after a safe endpoint migration.

    Only documented one-to-one field renames are applied.  No removed metadata
    is synthesized and no semantic fallback is attempted.
    """
    if not isinstance(payload, (dict, list)):
        return payload, None

    normalized = payload
    note = None
    if re.fullmatch(r"/playlists/[^/]+/tracks", original_path):
        normalized = _rename_playlist_entries(payload)
        note = "playlist entry item -> track"
    elif re.fullmatch(r"/playlists/[^/]+", original_path):
        normalized = _normalize_playlist_object(payload)
        note = "playlist object items -> tracks"
    elif original_path == "/me/playlists" and isinstance(payload, dict):
        normalized = dict(payload)
        entries = normalized.get("items")
        if isinstance(entries, list):
            normalized["items"] = [_normalize_playlist_object(x) for x in entries]
            note = "nested playlist object items -> tracks"
    elif original_path == "/search" and isinstance(payload, dict):
        # Search responses may contain playlist objects under playlists.items.
        # Normalize only that branch; album/artist/track responses are untouched.
        normalized = dict(payload)
        playlists = normalized.get("playlists")
        if isinstance(playlists, dict) and isinstance(playlists.get("items"), list):
            playlists = dict(playlists)
            playlists["items"] = [_normalize_playlist_object(x)
                                  for x in playlists["items"]]
            normalized["playlists"] = playlists
            note = "search playlist objects items -> tracks"
    return normalized, note


class SpotifyCompatibleResponse:
    """Transparent response proxy whose JSON/text expose a normalized payload."""
    def __init__(self, response: Any, payload: Any):
        self._response = response
        self._payload = payload
        self._text = json.dumps(payload, ensure_ascii=False, default=str)

    def json(self, **_kwargs):
        # Round-tripping returns an isolated ordinary JSON value without adding
        # a deepcopy dependency or exposing the cached object for mutation.
        return json.loads(self._text)

    @property
    def text(self):
        return self._text

    @property
    def content(self):
        return self._text.encode("utf-8")

    def __getattr__(self, name: str):
        return getattr(self._response, name)


def _normalize_response(response: Any, adapted: AdaptedRequest) -> tuple[Any, str | None]:
    if profile() != "dev2026" or adapted.original_method != "GET":
        return response, None
    try:
        payload = response.json()
    except Exception:
        return response, None
    normalized, note = normalize_response_payload(payload, adapted.original_path)
    if note is None or normalized == payload:
        return response, None
    return SpotifyCompatibleResponse(response, normalized), note


def _append_trace(entry: dict[str, Any]) -> None:
    path = os.environ.get("SECAT_SPOTIFY_TRACE_FILE")
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass



def _merge_search_payload(base: dict[str, Any], page: dict[str, Any]) -> int:
    """Append paging ``items`` branches from one Search response page."""
    largest_page = 0
    for key, value in page.items():
        if not (isinstance(value, dict) and isinstance(value.get("items"), list)):
            continue
        largest_page = max(largest_page, len(value["items"]))
        destination = base.get(key)
        if not (isinstance(destination, dict)
                and isinstance(destination.get("items"), list)):
            base[key] = deepcopy(value)
            continue
        destination["items"].extend(deepcopy(value["items"]))
        for metadata_key in ("next", "total"):
            if metadata_key in value:
                destination[metadata_key] = deepcopy(value[metadata_key])
    return largest_page


def _execute_adapted(adapted: AdaptedRequest) -> tuple[Any, int]:
    """Execute one adapted request, including transparent pagination.

    Search has a nested multi-type response and therefore uses its dedicated
    merger. Ordinary paged endpoints such as artist albums use a generic
    ``items`` merger. Both preserve the caller-visible historical limit while
    keeping one canonical benchmark trace entry.
    """
    if adapted.library_uris:
        first_response = None
        request_count = 0
        uris = list(adapted.library_uris)
        for start in range(0, len(uris), 40):
            chunk = uris[start:start + 40]
            chunk_kwargs = dict(adapted.kwargs)
            params = dict(chunk_kwargs.get("params") or {})
            params["uris"] = ",".join(chunk)
            chunk_kwargs["params"] = params
            response, attempts = _perform_spotify_request(
                adapted.method, adapted.url, **chunk_kwargs)
            request_count += attempts
            if first_response is None:
                first_response = response
            if not (isinstance(getattr(response, "status_code", None), int)
                    and 200 <= response.status_code < 300):
                return response, request_count
        assert first_response is not None
        return first_response, request_count

    target = adapted.search_target_limit
    if target and target > 10:
        first_response = None
        combined: dict[str, Any] | None = None
        request_count = 0
        fetched = 0
        while fetched < target:
            page_size = min(10, target - fetched)
            page_kwargs = dict(adapted.kwargs)
            page_params = dict(page_kwargs.get("params") or {})
            page_params["limit"] = page_size
            page_params["offset"] = adapted.search_start_offset + fetched
            page_kwargs["params"] = page_params
            response, attempts = _perform_spotify_request(
                adapted.method, adapted.url, **page_kwargs)
            request_count += attempts
            if first_response is None:
                first_response = response
            if not (isinstance(getattr(response, "status_code", None), int)
                    and 200 <= response.status_code < 300):
                return response, request_count
            try:
                payload = response.json()
            except Exception:
                return response, request_count
            if not isinstance(payload, dict):
                return response, request_count
            if combined is None:
                combined = deepcopy(payload)
                page_count = max(
                    (len(v.get("items", [])) for v in payload.values()
                     if isinstance(v, dict) and isinstance(v.get("items"), list)),
                    default=0,
                )
            else:
                page_count = _merge_search_payload(combined, payload)
            fetched += page_size
            if page_count < page_size:
                break

        assert first_response is not None
        if combined is None:
            return first_response, request_count
        for value in combined.values():
            if isinstance(value, dict) and isinstance(value.get("items"), list):
                value["limit"] = target
                value["offset"] = adapted.search_start_offset
        return SpotifyCompatibleResponse(first_response, combined), request_count

    # Generic one-list paging used for endpoint-specific 2026 caps.
    target = adapted.paging_target_limit
    cap = int(adapted.paging_page_cap or 0)
    if target and cap > 0 and target > cap:
        first_response = None
        combined: dict[str, Any] | None = None
        request_count = 0
        fetched = 0
        while fetched < target:
            page_size = min(cap, target - fetched)
            page_kwargs = dict(adapted.kwargs)
            page_params = dict(page_kwargs.get("params") or {})
            page_params["limit"] = page_size
            page_params["offset"] = adapted.paging_start_offset + fetched
            page_kwargs["params"] = page_params
            response, attempts = _perform_spotify_request(
                adapted.method, adapted.url, **page_kwargs)
            request_count += attempts
            if first_response is None:
                first_response = response
            if not (isinstance(getattr(response, "status_code", None), int)
                    and 200 <= response.status_code < 300):
                return response, request_count
            try:
                payload = response.json()
            except Exception:
                return response, request_count
            if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
                return response, request_count
            if combined is None:
                combined = deepcopy(payload)
            else:
                combined.setdefault("items", []).extend(deepcopy(payload.get("items") or []))
                for metadata_key in ("next", "total"):
                    if metadata_key in payload:
                        combined[metadata_key] = deepcopy(payload[metadata_key])
            page_count = len(payload.get("items") or [])
            fetched += page_size
            if page_count < page_size:
                break

        assert first_response is not None
        if combined is None:
            return first_response, request_count
        combined["limit"] = target
        combined["offset"] = adapted.paging_start_offset
        return SpotifyCompatibleResponse(first_response, combined), request_count

    return _perform_spotify_request(adapted.method, adapted.url, **adapted.kwargs)


def request(method: str, url: str, _secat_access_token: str | None = None, **kwargs):
    original_kwargs = dict(kwargs)
    adapted = _prepare(method, url, original_kwargs, access_token=_secat_access_token)
    raw_response, effective_requests = _execute_adapted(adapted)
    auth_refreshed = False

    # Spotify access tokens are short-lived. A long benchmark must not collapse
    # halfway through because the token expired. Refresh once on a genuine 401,
    # then rebuild the request with the new trusted credential and retry.
    if (str(url).startswith(BASE_URL)
            and getattr(raw_response, "status_code", None) == 401):
        fresh = _refresh_access_token()
        if fresh:
            retry_kwargs = dict(original_kwargs)
            retry_headers = dict(retry_kwargs.get("headers") or {})
            retry_headers.pop("Authorization", None)
            retry_kwargs["headers"] = retry_headers
            adapted = _prepare(method, url, retry_kwargs, access_token=fresh)
            raw_response, retry_count = _execute_adapted(adapted)
            effective_requests += retry_count
            auth_refreshed = True

    response, response_note = _normalize_response(raw_response, adapted)
    combined_note = adapted.note
    if response_note:
        combined_note = "; ".join(x for x in (adapted.note, response_note) if x)
    if auth_refreshed:
        combined_note = "; ".join(x for x in (combined_note, "OAuth access token refreshed after 401") if x)
    _append_trace({
        "method": adapted.original_method,
        "endpoint": adapted.original_path,
        "effective_method": adapted.method,
        "effective_url": adapted.url,
        "status_code": getattr(raw_response, "status_code", None),
        "adapted": bool(adapted.adapted or response_note),
        "adaptation_note": combined_note,
        "effective_requests": effective_requests,
        "search_target_limit": adapted.search_target_limit,
        "library_uri_count": len(adapted.library_uris or []),
        "auth_refreshed": auth_refreshed,
        "rate_limit": getattr(raw_response, "secat_rate_limit", None),
    })
    try:
        response.secat_api_adaptation = {
            "adapted": bool(adapted.adapted or response_note), "note": combined_note,
            "original_method": adapted.original_method,
            "original_path": adapted.original_path,
            "effective_url": adapted.url,
            "effective_requests": effective_requests,
            "search_target_limit": adapted.search_target_limit,
            "library_uri_count": len(adapted.library_uris or []),
            "auth_refreshed": auth_refreshed,
            "rate_limit": getattr(raw_response, "secat_rate_limit", None),
        }
    except Exception:
        pass
    return response


class SpotifyRequestsProxy:
    """Module-like requests proxy used inside agent execution environments.

    The access token is captured by trusted startup so the raw environment
    variable can be removed before generated code executes.
    """
    exceptions = _requests.exceptions
    RequestException = _requests.RequestException
    codes = _requests.codes
    Response = _requests.Response

    def __init__(self, access_token: str | None = None):
        self._access_token = (os.environ.get("SPOTIFY_ACCESS_TOKEN", "")
                              if access_token is None else str(access_token))

    def request(self, method, url, **kwargs):
        response = request(method, url, _secat_access_token=self._access_token, **kwargs)
        if getattr(response, "status_code", None) == 429:
            _retry_after, _reason, _quota_exceeded = _rate_limit_details(response)
            if _quota_exceeded:
                raise _rate_limit_error(response)
        return response
    def get(self, url, **kwargs): return self.request("GET", url, **kwargs)
    def post(self, url, **kwargs): return self.request("POST", url, **kwargs)
    def put(self, url, **kwargs): return self.request("PUT", url, **kwargs)
    def delete(self, url, **kwargs): return self.request("DELETE", url, **kwargs)
    def patch(self, url, **kwargs): return self.request("PATCH", url, **kwargs)

    class _Session:
        def __init__(self, owner):
            self._owner = owner
            self.headers = {}
        def request(self, method, url, **kwargs):
            headers = dict(self.headers); headers.update(kwargs.pop("headers", {}) or {})
            return self._owner.request(method, url, headers=headers, **kwargs)
        def get(self, url, **kwargs): return self.request("GET", url, **kwargs)
        def post(self, url, **kwargs): return self.request("POST", url, **kwargs)
        def put(self, url, **kwargs): return self.request("PUT", url, **kwargs)
        def delete(self, url, **kwargs): return self.request("DELETE", url, **kwargs)
        def patch(self, url, **kwargs): return self.request("PATCH", url, **kwargs)

    def Session(self):
        return self._Session(self)


class SpotifyRequestsWrapper:
    """Small ToolCoder-compatible wrapper with ``get/post/put/delete`` methods."""
    def __init__(self, headers: dict | None = None, *, raise_on_rate_limit: bool = False):
        self.headers = dict(headers or {})
        self.raise_on_rate_limit = bool(raise_on_rate_limit)
    def _call(self, method, url, **kwargs):
        headers = dict(self.headers); headers.update(kwargs.pop("headers", {}) or {})
        # The original ToolCoder prompts call this field `data` while describing
        # a JSON requestBody. Convert dicts to JSON for Spotify's Web API.
        if isinstance(kwargs.get("data"), dict) and "json" not in kwargs:
            kwargs["json"] = kwargs.pop("data")
        response = request(method, url, headers=headers, **kwargs)
        if getattr(response, "status_code", None) == 429:
            _retry_after, _reason, _quota_exceeded = _rate_limit_details(response)
            if _quota_exceeded or self.raise_on_rate_limit:
                raise _rate_limit_error(response)
        return response
    def get(self, url, **kwargs): return self._call("GET", url, **kwargs)
    def post(self, url, **kwargs): return self._call("POST", url, **kwargs)
    def put(self, url, **kwargs): return self._call("PUT", url, **kwargs)
    def delete(self, url, **kwargs): return self._call("DELETE", url, **kwargs)
    def patch(self, url, **kwargs): return self._call("PATCH", url, **kwargs)


def install_requests_compat():
    """Return a proxy that captures the OAuth token during trusted startup."""
    return SpotifyRequestsProxy(access_token=os.environ.get("SPOTIFY_ACCESS_TOKEN", ""))


def clear_trace(path: str | None = None) -> None:
    """Remove the current task trace (used after trusted fixture setup)."""
    path = path or os.environ.get("SECAT_SPOTIFY_TRACE_FILE")
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


def read_trace(path: str | None = None) -> list[dict[str, Any]]:
    path = path or os.environ.get("SECAT_SPOTIFY_TRACE_FILE")
    if not path or not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            try:
                value = json.loads(line)
                if isinstance(value, dict): out.append(value)
            except Exception:
                pass
    return out


def _path_matches(actual: str, template: str) -> bool:
    chunks, cursor = [], 0
    for match in re.finditer(r"\{[^{}]+\}", template or ""):
        chunks.append(re.escape(template[cursor:match.start()]))
        chunks.append(r"[^/]+")
        cursor = match.end()
    chunks.append(re.escape((template or "")[cursor:]))
    return bool(re.fullmatch("".join(chunks).rstrip("/") + r"/?", (actual or "").rstrip("/")))


def evaluate_trace(task: dict, trace: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Shared method/path/status scorer used by all three Spotify agents."""
    trace = list(trace if trace is not None else read_trace())
    required = list(task.get("solution_steps") or [])
    cursor, matched = 0, []
    for step in required:
        found = None
        while cursor < len(trace):
            call = trace[cursor]; cursor += 1
            if (str(call.get("method", "GET")).upper() == str(step.get("method", "GET")).upper()
                    and _path_matches(call.get("endpoint", ""), step.get("path", ""))):
                found = call; break
        if found is None: break
        matched.append(found)
    complete = bool(required) and len(matched) == len(required)
    statuses_ok = complete and all(isinstance(x.get("status_code"), int) and
                                   200 <= x["status_code"] < 300 for x in matched)
    return {"required_steps": required, "trace_calls": len(trace),
            "matched_steps": len(matched), "path_sequence_complete": complete,
            "http_success": statuses_ok, "matched_calls": matched,
            "score": bool(complete and statuses_ok)}


def _identifier_tokens(value: Any) -> set[str]:
    """Normalize Spotify ids/URIs into comparable, non-route tokens."""
    route_words = {
        "spotify", "playlist", "playlists", "track", "tracks", "album", "albums",
        "artist", "artists", "follower", "followers", "following", "player",
        "queue", "users", "shows", "episodes", "audiobooks", "chapters",
    }
    found: set[str] = set()
    if isinstance(value, (str, int)):
        text = str(value)
        for token in re.findall(r"[A-Za-z0-9_-]+", text):
            if len(token) >= 6 and token.lower() not in route_words:
                found.add(token)
    return found


def _walk_identifiers(value: Any, key: str = "") -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for child_key, child in value.items():
            found.update(_walk_identifiers(child, str(child_key).lower()))
    elif isinstance(value, (list, tuple, set)):
        for child in value:
            found.update(_walk_identifiers(child, key))
    elif (key in {"id", "ids", "uri", "uris", "href"} or
          key.endswith("_id") or key.endswith("_ids") or
          key.endswith("_uri") or key.endswith("_uris")):
        found.update(_identifier_tokens(value))
    return found


def _action_identifiers(call: dict[str, Any], plan_step: dict[str, Any] | None = None) -> list[str]:
    """Extract concrete target identifiers from an observed write request.

    Path identifiers are determined from the validated plan template rather than
    from arbitrary long path words.  This prevents static segments such as
    ``volume`` or an adapter's ``library`` route from being mistaken for entity
    identifiers while still recovering concrete values for placeholders such as
    ``{playlist_id}``.  Query/body identifiers keep the existing typed-key rules.
    """
    found: set[str] = set()
    concrete = str(call.get("endpoint") or call.get("effective_endpoint") or "").split("?", 1)[0]
    template = str((plan_step or {}).get("endpoint") or "").split("?", 1)[0]

    def parts(path: str) -> list[str]:
        out = [x for x in path.split("/") if x]
        if out and out[0].casefold() == "v1":
            out = out[1:]
        return out

    concrete_parts = parts(concrete)
    template_parts = parts(template)
    if template_parts and len(concrete_parts) >= len(template_parts):
        # Align on the suffix so absolute/relative traces and an optional /v1
        # prefix are handled consistently.
        concrete_parts = concrete_parts[-len(template_parts):]
        for actual, declared in zip(concrete_parts, template_parts):
            if declared.startswith("{") and declared.endswith("}"):
                found.update(_identifier_tokens(actual))
    elif not template_parts:
        # Compatibility fallback for historical ledgers without a plan step.
        route_words = {
            "spotify", "playlist", "playlists", "track", "tracks", "album", "albums",
            "artist", "artists", "follower", "followers", "following", "player",
            "queue", "users", "shows", "episodes", "audiobooks", "chapters",
            "volume", "repeat", "next", "pause", "play", "library", "items", "me",
        }
        for segment in concrete_parts:
            tokens = _identifier_tokens(segment)
            found.update(x for x in tokens if x.casefold() not in route_words)

    found.update(_walk_identifiers(call.get("params") or {}))
    found.update(_walk_identifiers(call.get("request_body")))
    return sorted(found)


def _observation_identifiers(observation: dict[str, Any]) -> set[str]:
    found = _identifier_tokens(observation.get("record_id"))
    found.update(_walk_identifiers(observation.get("fields") or {}))
    return found


def action_certificate(ledger: Any, plan: dict[str, Any] | None = None) -> dict[str, Any]:
    """Gold-free certificate for state-changing Spotify actions.

    Every observed write must succeed, have a citable ``action_result`` from the
    same call, and target identifiers that were established by an earlier
    observation or explicitly declared as a validated literal in that exact plan
    step. Writes without entity identifiers (pause, skip, create-by-name) are
    certified by their same-call action result. Benchmark ``solution_steps``
    are deliberately absent and remain a post-run oracle metric only.
    """
    calls = list(getattr(ledger, "api_calls", []) or [])
    observations = list(getattr(ledger, "observations", []) or [])
    writes = [c for c in calls
              if str(c.get("method", "GET")).upper() in WRITE_METHODS]
    failed = [c for c in writes if not (isinstance(c.get("status_code"), int)
                                         and 200 <= c["status_code"] < 300)]
    action_obs_by_call: dict[str, list[dict[str, Any]]] = {}
    obs_by_call: dict[str, list[dict[str, Any]]] = {}
    for obs in observations:
        cid = str(obs.get("call_id") or "")
        if cid:
            obs_by_call.setdefault(cid, []).append(obs)
        if obs.get("kind") == "action_result" and cid:
            action_obs_by_call.setdefault(cid, []).append(obs)

    step_literal_ids: dict[str, set[str]] = {}
    plan_steps_by_id: dict[str, dict[str, Any]] = {}
    for step in (plan or {}).get("steps") or []:
        if isinstance(step, dict) and step.get("id"):
            plan_steps_by_id[str(step.get("id"))] = step
        sid = str(step.get("id") or "")
        found: set[str] = set()
        found.update(_walk_identifiers(step.get("path_literals") or {}))
        found.update(_walk_identifiers(step.get("query_literals") or {}))
        found.update(_walk_identifiers(step.get("body_literals") or {}))
        if sid and found:
            step_literal_ids[sid] = found

    # Reuse the generic execution auditor's exact request-contract result. Live
    # requests already crossed the trusted guard against full producer payloads;
    # a later compact observation view must not create a contradictory, weaker
    # provenance verdict.
    audited_valid_calls: set[str] = set()
    audited_valid_by_step: dict[str, set[str]] = {}
    if plan:
        try:
            from utils.plan_execution_audit import audit_execution
            _audit = audit_execution(plan, ledger)
            for _sid, _status in ((_audit or {}).get("step_status") or {}).items():
                _ids = {str(x) for x in (_status.get("valid_call_ids") or []) if str(x)}
                audited_valid_by_step[str(_sid)] = _ids
                audited_valid_calls.update(_ids)
        except Exception:
            # Historical/imported ledgers retain the existing observation-based
            # reconstruction below if the generic auditor is unavailable.
            audited_valid_calls = set()
            audited_valid_by_step = {}

    call_order = {str(call.get("call_id") or ""): i for i, call in enumerate(calls)}
    missing_action_results: list[str] = []
    lineage: list[dict[str, Any]] = []
    cited: list[str] = []
    lineage_failures: list[dict[str, Any]] = []
    for call in writes:
        cid = str(call.get("call_id") or "")
        obs_rows = action_obs_by_call.get(cid, [])
        if not obs_rows:
            missing_action_results.append(cid)
        cited.extend(str(o.get("obs_id")) for o in obs_rows if o.get("obs_id"))
        plan_step = plan_steps_by_id.get(str(call.get("plan_step_id") or ""))
        requested_ids = set(_action_identifiers(call, plan_step))
        available_ids: set[str] = set()
        current_order = call_order.get(cid, 10**9)
        for prior_call_id, prior_rows in obs_by_call.items():
            if call_order.get(prior_call_id, 10**9) >= current_order:
                continue
            for prior in prior_rows:
                available_ids.update(_observation_identifiers(prior))
        declared_literal_ids = set(step_literal_ids.get(str(call.get("plan_step_id") or "")) or set())
        unresolved = sorted(requested_ids - available_ids - declared_literal_ids)
        request_contract_certified = cid in audited_valid_calls
        # A valid generic execution-audit call proves the exact path/query/body
        # bindings against trusted producer lineage. Prefer that stronger proof
        # over a lossy scan of compact observations.
        lineage_ok = bool(obs_rows) and (request_contract_certified or not unresolved)
        item = {
            "call_id": cid,
            "method": str(call.get("method", "GET")).upper(),
            "endpoint": call.get("endpoint"),
            "request_identifiers": sorted(requested_ids),
            "available_prior_identifiers": sorted(available_ids),
            "declared_literal_identifiers": sorted(declared_literal_ids),
            "unresolved_identifiers": unresolved,
            "action_result_observation_ids": [o.get("obs_id") for o in obs_rows],
            "same_call_provenance": bool(obs_rows),
            "prior_identifier_lineage": not unresolved,
            "request_contract_certified": request_contract_certified,
            "lineage_ok": lineage_ok,
        }
        lineage.append(item)
        if not lineage_ok:
            lineage_failures.append(item)

    obligations = [dict(x) for x in ((plan or {}).get("action_obligations") or [])
                   if isinstance(x, dict)]
    missing_obligations: list[dict[str, Any]] = []
    for obligation in obligations:
        sid = str(obligation.get("step_id") or "")
        valid_ids = set(audited_valid_by_step.get(sid) or set())
        satisfied = False
        for cid0 in valid_ids:
            call0 = next((c for c in writes if str(c.get("call_id") or "") == cid0), None)
            if call0 is not None and action_obs_by_call.get(cid0):
                satisfied = True; break
        if not satisfied:
            missing_obligations.append(obligation)

    complete = bool(writes) and not missing_action_results and not missing_obligations
    lineage_ok = bool(writes) and not lineage_failures
    accepted = complete and not failed and lineage_ok
    return {
        "accepted": bool(accepted),
        "observed_write_calls": writes,
        "matched_calls": writes,  # compatibility alias; not benchmark matching
        "complete": bool(complete),
        "failed_calls": failed,
        "missing_action_result_calls": missing_action_results,
        "action_obligations": obligations,
        "missing_action_obligations": missing_obligations,
        "cited_observation_ids": list(dict.fromkeys(cited)),
        "identifier_lineage": lineage,
        "identifier_lineage_ok": lineage_ok,
        "lineage_failures": lineage_failures,
        "verification_status": "accepted" if accepted else "action_gap",
    }


def _select_active_unrestricted_device(devices: Any) -> dict[str, Any] | None:
    if not isinstance(devices, list):
        return None
    return next(
        (d for d in devices if isinstance(d, dict) and d.get("is_active")
         and not d.get("is_restricted")),
        None,
    )


def _select_transferable_unrestricted_device(devices: Any) -> dict[str, Any] | None:
    """Return an inactive Spotify Connect device that can be activated safely.

    Transfer Playback requires a concrete device id.  Restricted devices cannot
    accept Web API commands, and private-session devices are intentionally left
    alone by the benchmark fixture.
    """
    if not isinstance(devices, list):
        return None
    candidates = [
        d for d in devices
        if isinstance(d, dict)
        and not d.get("is_active")
        and not d.get("is_restricted")
        and not d.get("is_private_session")
        and str(d.get("id") or "").strip()
    ]
    return candidates[0] if candidates else None


def _device_summary(devices: Any) -> list[dict[str, Any]]:
    if not isinstance(devices, list):
        return []
    return [
        {
            "id": str(d.get("id") or ""),
            "name": str(d.get("name") or ""),
            "type": str(d.get("type") or ""),
            "is_active": bool(d.get("is_active")),
            "is_restricted": bool(d.get("is_restricted")),
            "is_private_session": bool(d.get("is_private_session")),
        }
        for d in devices if isinstance(d, dict)
    ]


def _ensure_active_unrestricted_device(api: Any) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Find or activate a controllable Spotify Connect device for fixture reset.

    A Spotify client can remain visible through ``GET /me/player/devices`` while
    no device is marked active.  In that state the old fixture aborted even
    though Spotify's Transfer Playback endpoint can activate the available
    device.  This helper performs that bounded recovery only in trusted fixture
    setup; agent-visible benchmark execution is unchanged.
    """
    retries = max(1, int(os.environ.get("SPOTIFY_DEVICE_ACTIVATION_RETRIES", "3")))
    settle = max(0.0, float(os.environ.get("SPOTIFY_DEVICE_ACTIVATION_SETTLE_SEC", "0.75")))

    response = api.get(BASE_URL + "/me/player/devices")
    response.raise_for_status()
    devices = (response.json() or {}).get("devices", [])
    active = _select_active_unrestricted_device(devices)
    meta: dict[str, Any] = {
        "attempted": False,
        "transferred": False,
        "selected_device_id": str((active or {}).get("id") or ""),
        "available_devices": _device_summary(devices),
    }
    if active:
        return active, meta

    candidate = _select_transferable_unrestricted_device(devices)
    if not candidate:
        return None, meta

    candidate_id = str(candidate.get("id") or "")
    meta.update({
        "attempted": True,
        "selected_device_id": candidate_id,
        "selected_device_name": str(candidate.get("name") or ""),
    })
    transfer = api.put(
        BASE_URL + "/me/player",
        json={"device_ids": [candidate_id], "play": True},
    )
    meta["transfer_status"] = int(getattr(transfer, "status_code", 0) or 0)
    if not (200 <= meta["transfer_status"] < 300):
        meta["transfer_error"] = str(getattr(transfer, "text", ""))[:300]
        return None, meta
    meta["transferred"] = True

    for attempt in range(1, retries + 1):
        if settle:
            time.sleep(settle)
        response = api.get(BASE_URL + "/me/player/devices")
        response.raise_for_status()
        devices = (response.json() or {}).get("devices", [])
        active = _select_active_unrestricted_device(devices)
        meta["activation_verify_attempts"] = attempt
        meta["available_devices_after_transfer"] = _device_summary(devices)
        if active:
            meta["selected_device_id"] = str(active.get("id") or "")
            return active, meta
    return None, meta


def _device_target_params(device: dict[str, Any] | None) -> tuple[str, dict[str, str]]:
    """Return displayable device id and safe player-command query params.

    Spotify device ids are nullable/not guaranteed.  Omitting ``device_id``
    targets the currently active device, which is the correct fallback.
    """
    device_id = str((device or {}).get("id") or "")
    return device_id, ({"device_id": device_id} if device_id else {})


def _fixture_cache_path() -> Path | None:
    raw = os.environ.get("SECAT_SPOTIFY_FIXTURE_CACHE", "").strip()
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        root = Path(os.environ.get("SECAT_PROJECT_ROOT") or Path(__file__).resolve().parents[1])
        path = root / path
    return path


def _fixture_cache_key(q: str, typ: str, limit: int) -> str:
    return json.dumps(
        [1, profile(), str(typ), int(limit), str(q)],
        ensure_ascii=False, separators=(",", ":"))


def _minimal_fixture_item(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    item = {key: value.get(key) for key in ("id", "uri", "name", "type")
            if value.get(key) not in (None, "")}
    return item if item.get("id") or item.get("uri") else None


def _load_fixture_cache(path: Path | None) -> None:
    if path is None:
        return
    marker = str(path.resolve())
    with _FIXTURE_CACHE_LOCK:
        if marker in _FIXTURE_CACHE_LOADED_PATHS:
            return
        _FIXTURE_CACHE_LOADED_PATHS.add(marker)
        try:
            if not path.is_file() or path.stat().st_size > 1_000_000:
                return
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or payload.get("version") != 1:
                return
            entries = payload.get("entries") or {}
            if not isinstance(entries, dict):
                return
            for key, values in entries.items():
                if not isinstance(key, str) or not isinstance(values, list):
                    continue
                cleaned = [item for item in
                           (_minimal_fixture_item(value) for value in values)
                           if item is not None]
                if cleaned:
                    _FIXTURE_SEARCH_CACHE[key] = cleaned
        except Exception:
            # A cache is only an optimization. Corruption or filesystem policy
            # must fall back to live lookup rather than block fixture setup.
            return


def _save_fixture_cache(path: Path | None) -> None:
    if path is None:
        return
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _FIXTURE_CACHE_LOCK:
            payload = {"version": 1, "entries": _FIXTURE_SEARCH_CACHE}
            tmp.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                encoding="utf-8")
            os.replace(tmp, path)
    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def _fixture_search(api: Any, q: str, typ: str, limit: int = 10
                    ) -> tuple[list[dict[str, Any]], bool]:
    """Resolve one deterministic fixture seed, reusing only valid catalog IDs."""
    path = _fixture_cache_path()
    _load_fixture_cache(path)
    key = _fixture_cache_key(q, typ, limit)
    with _FIXTURE_CACHE_LOCK:
        cached = deepcopy(_FIXTURE_SEARCH_CACHE.get(key) or [])
    if cached:
        return cached, True
    response = api.get(
        BASE_URL + "/search", params={"q": q, "type": typ, "limit": limit})
    response.raise_for_status()
    payload = response.json() or {}
    values = (payload.get(typ + "s") or {}).get("items", [])
    if not isinstance(values, list):
        values = []
    cleaned = [item for item in
               (_minimal_fixture_item(value) for value in values)
               if item is not None]
    if cleaned:
        with _FIXTURE_CACHE_LOCK:
            _FIXTURE_SEARCH_CACHE[key] = deepcopy(cleaned)
        _save_fixture_cache(path)
    return cleaned, False


def reset_fixture() -> dict:
    """Destructively recreate the original RestBench-style account fixture.

    This is intentionally opt-in twice: writes must be enabled and
    ``ALLOW_SPOTIFY_RESET=YES`` must be set.  The function uses search results
    rather than artist top-tracks so setup also works under the dev2026 profile.
    """
    if not writes_allowed() or os.environ.get("ALLOW_SPOTIFY_RESET", "").upper() != "YES":
        raise SpotifyWriteBlocked(
            "fixture reset is destructive; require --allow-spotify-writes and "
            "ALLOW_SPOTIFY_RESET=YES on a disposable account")
    api = SpotifyRequestsWrapper(raise_on_rate_limit=True)
    report: dict[str, Any] = {
        "removed": {}, "created": {}, "warnings": [],
        "seed_cache": {"hits": 0, "misses": 0,
                       "persistent": _fixture_cache_path() is not None},
    }

    def items(path, key="items"):
        out = []
        url = BASE_URL + path
        params = {"limit": 50}
        seen = set()
        while url and url not in seen:
            seen.add(url)
            r = api.get(url, params=params if url == BASE_URL + path else None)
            r.raise_for_status()
            payload = r.json() or {}
            page = payload.get(key, [])
            if isinstance(page, list):
                out.extend(page)
            url = payload.get("next") if isinstance(payload, dict) else None
            params = None
        return out
    def search(q, typ, limit=10):
        found, cached = _fixture_search(api, q, typ, limit)
        key = "hits" if cached else "misses"
        report["seed_cache"][key] += 1
        return found

    me = api.get(BASE_URL + "/me"); me.raise_for_status(); user_id = me.json()["id"]
    playlists = items("/me/playlists")
    for p in playlists:
        api.delete(BASE_URL + f"/playlists/{p['id']}/followers").raise_for_status()
    report["removed"]["playlists"] = len(playlists)

    saved_tracks = [x.get("track", {}).get("id") for x in items("/me/tracks")]
    saved_tracks = [x for x in saved_tracks if x]
    if saved_tracks:
        api.delete(BASE_URL + "/me/tracks", params={"ids": ",".join(saved_tracks)}).raise_for_status()
    report["removed"]["tracks"] = len(saved_tracks)

    saved_albums = [x.get("album", {}).get("id") for x in items("/me/albums")]
    saved_albums = [x for x in saved_albums if x]
    if saved_albums:
        api.delete(BASE_URL + "/me/albums", params={"ids": ",".join(saved_albums)}).raise_for_status()
    report["removed"]["albums"] = len(saved_albums)

    followed = []
    follow_url = BASE_URL + "/me/following"
    follow_params = {"type": "artist", "limit": 50}
    seen_follow = set()
    while follow_url and follow_url not in seen_follow:
        seen_follow.add(follow_url)
        fr = api.get(follow_url, params=follow_params if follow_url == BASE_URL + "/me/following" else None)
        fr.raise_for_status()
        artists_page = (fr.json() or {}).get("artists", {}) or {}
        followed.extend(x["id"] for x in artists_page.get("items", []) if x.get("id"))
        follow_url = artists_page.get("next")
        follow_params = None
    if followed:
        api.delete(BASE_URL + "/me/following",
                   params={"type": "artist", "ids": ",".join(followed)}).raise_for_status()
    report["removed"]["followed_artists"] = len(followed)

    artists = {}
    for name in ("Lana Del Rey", "Whitney Houston", "The Beatles"):
        found = search(f'artist:"{name}"', "artist", 1)
        if not found: raise SpotifyBenchmarkError(f"fixture artist not found: {name}")
        artists[name] = found[0]["id"]
    api.put(BASE_URL + "/me/following",
            params={"type": "artist", "ids": ",".join(artists.values())}).raise_for_status()
    report["created"]["followed_artists"] = list(artists)

    track_queries = {
        "Lana Del Rey": ["Video Games", "Summertime Sadness", "Born To Die"],
        "Whitney Houston": ["I Wanna Dance with Somebody", "I Will Always Love You", "How Will I Know"],
        "The Beatles": ["Here Comes The Sun", "Come Together", "Let It Be"],
    }
    tracks: dict[str, list[str]] = {}
    for artist, titles in track_queries.items():
        # Reproduce the supplied ToolCoder fixture exactly when the historical
        # top-tracks endpoint exists. The restricted profile uses deterministic
        # named-track searches because top-tracks has no replacement.
        if profile() == "legacy":
            top = api.get(BASE_URL + f"/artists/{artists[artist]}/top-tracks",
                          params={"market": "US"})
            top.raise_for_status()
            tracks[artist] = [x.get("uri") for x in (top.json() or {}).get("tracks", [])[:3]
                              if x.get("uri")]
        else:
            tracks[artist] = []
            for title in titles:
                found = search(f'track:"{title}" artist:"{artist}"', "track", 1)
                if found and found[0].get("uri"):
                    tracks[artist].append(found[0]["uri"])
        if len(tracks[artist]) < 3:
            raise SpotifyBenchmarkError(
                f"fixture could not resolve three tracks for {artist}: {tracks[artist]}")
    library_uris = tracks["Lana Del Rey"] + tracks["Whitney Houston"]
    if library_uris:
        api.put(BASE_URL + "/me/tracks", params={"ids": ",".join(u.rsplit(":",1)[-1] for u in library_uris)}).raise_for_status()
    report["created"]["saved_tracks"] = library_uris

    album_uris = []
    for q in ('album:"Born To Die" artist:"Lana Del Rey"',
              'album:"reputation" artist:"Taylor Swift"'):
        found = search(q, "album", 1)
        if found and found[0].get("uri"):
            album_uris.append(found[0]["uri"])
    if len(album_uris) != 2:
        raise SpotifyBenchmarkError(f"fixture albums unresolved: {album_uris}")
    api.put(BASE_URL + "/me/albums",
            params={"ids": ",".join(u.rsplit(":",1)[-1] for u in album_uris)}).raise_for_status()
    report["created"]["saved_albums"] = album_uris

    for name, artist in (("My R&B", "Whitney Houston"), ("My Rock", "The Beatles")):
        r = api.post(BASE_URL + f"/users/{user_id}/playlists",
                     json={"name": name, "public": False})
        r.raise_for_status(); pid = r.json()["id"]
        if tracks[artist]:
            api.post(BASE_URL + f"/playlists/{pid}/tracks",
                     json={"uris": tracks[artist]}).raise_for_status()
        report["created"].setdefault("playlists", {})[name] = pid

    # The supplied RestBench/ToolCoder fixture expects the song "Born To Die"
    # to be the current playback item.  Starting an album context is not strong
    # enough on current Spotify Connect: a player may acknowledge the command
    # while retaining its previous context.  Resolve the exact seeded track,
    # address the active unrestricted device explicitly, and verify the track ID.
    born_to_die_uri = tracks["Lana Del Rey"][2]
    expected_playback_track = born_to_die_uri.rsplit(":", 1)[-1]

    active_device, device_activation = _ensure_active_unrestricted_device(api)
    report["device_activation"] = device_activation
    if not active_device:
        raise SpotifyBenchmarkError(
            "fixture could not find or activate an unrestricted Spotify device; "
            f"available_devices={device_activation.get('available_devices_after_transfer') or device_activation.get('available_devices') or []}. "
            "Open Spotify on a controllable device once, then rerun; the fixture will "
            "automatically reactivate an available inactive device on later resets.")
    # Spotify documents device IDs as nullable/not guaranteed.  When absent,
    # /me/player/play is allowed to target the currently active device by
    # omitting device_id entirely.
    device_id, device_params = _device_target_params(active_device)

    def _start_fixture_track():
        response = api.put(
            BASE_URL + "/me/player/play",
            params=device_params,
            json={"uris": [born_to_die_uri], "position_ms": 0},
        )
        if not (200 <= response.status_code < 300):
            raise SpotifyBenchmarkError(
                "fixture could not start Born To Die on the active Spotify "
                f"device (HTTP {response.status_code}: {response.text[:300]})")
        return response

    _start_fixture_track()
    time.sleep(float(os.environ.get("SPOTIFY_FIXTURE_SETTLE_SEC", "1.0")))
    report["created"]["current_playback"] = born_to_die_uri
    report["created"]["playback_device_id"] = device_id

    # Read the fixture back through the same public API before declaring the
    # benchmark state canonical. A successful 2xx write is not sufficient.
    expected_track_ids = {u.rsplit(":", 1)[-1] for u in library_uris}
    expected_album_ids = {u.rsplit(":", 1)[-1] for u in album_uris}
    expected_artist_ids = set(artists.values())
    expected_playlist_names = {"My R&B", "My Rock"}
    verify_retries = max(1, int(os.environ.get("SPOTIFY_FIXTURE_VERIFY_RETRIES", "5")))
    verify_sleep = max(0.0, float(os.environ.get("SPOTIFY_FIXTURE_VERIFY_SLEEP_SEC", "0.75")))
    verification = {}
    last_errors = []
    for attempt in range(1, verify_retries + 1):
        observed_tracks = {
            (x.get("track") or {}).get("id") for x in items("/me/tracks")
            if isinstance(x, dict) and (x.get("track") or {}).get("id")
        }
        observed_albums = {
            (x.get("album") or {}).get("id") for x in items("/me/albums")
            if isinstance(x, dict) and (x.get("album") or {}).get("id")
        }
        observed_playlists = {
            str(x.get("name") or "") for x in items("/me/playlists")
            if isinstance(x, dict) and x.get("name")
        }
        fr = api.get(BASE_URL + "/me/following", params={"type": "artist", "limit": 50})
        fr.raise_for_status()
        observed_artists = {
            str(x.get("id")) for x in ((fr.json() or {}).get("artists", {}) or {}).get("items", [])
            if isinstance(x, dict) and x.get("id")
        }
        now = api.get(BASE_URL + "/me/player/currently-playing")
        playback_track = ""
        playback_album = ""
        playback_is_playing = False
        if now.status_code == 200:
            try:
                now_payload = now.json() or {}
                now_item = now_payload.get("item") or {}
                playback_track = str(now_item.get("id") or "")
                playback_album = str((now_item.get("album") or {}).get("id") or "")
                playback_is_playing = bool(now_payload.get("is_playing", True))
            except Exception:
                playback_track = ""
                playback_album = ""
                playback_is_playing = False

        last_errors = []
        if observed_tracks != expected_track_ids:
            last_errors.append(f"saved tracks mismatch expected={sorted(expected_track_ids)} observed={sorted(observed_tracks)}")
        if observed_albums != expected_album_ids:
            last_errors.append(f"saved albums mismatch expected={sorted(expected_album_ids)} observed={sorted(observed_albums)}")
        if observed_artists != expected_artist_ids:
            last_errors.append(f"followed artists mismatch expected={sorted(expected_artist_ids)} observed={sorted(observed_artists)}")
        if observed_playlists != expected_playlist_names:
            last_errors.append(f"playlists mismatch expected={sorted(expected_playlist_names)} observed={sorted(observed_playlists)}")
        if playback_track != expected_playback_track or not playback_is_playing:
            last_errors.append(
                f"current playback mismatch expected track={expected_playback_track!r} "
                f"observed_track={playback_track!r} observed_album={playback_album!r} "
                f"is_playing={playback_is_playing!r} HTTP={getattr(now, 'status_code', None)}")

        verification = {
            "attempt": attempt,
            "saved_tracks": sorted(observed_tracks),
            "saved_albums": sorted(observed_albums),
            "followed_artists": sorted(observed_artists),
            "playlists": sorted(observed_playlists),
            "current_playback_track": playback_track,
            "current_playback_album": playback_album,
            "current_playback_is_playing": playback_is_playing,
            "playback_device_id": device_id,
            "ok": not last_errors,
        }
        if not last_errors:
            break
        if attempt < verify_retries:
            # Spotify Connect can acknowledge a player command before the target
            # device actually switches. Re-issue the same deterministic command
            # rather than merely waiting on stale playback state.
            _start_fixture_track()
            time.sleep(verify_sleep)
    report["verification"] = verification
    if last_errors:
        raise SpotifyBenchmarkError(
            "fixture read-back verification failed after "
            f"{verify_retries} attempt(s): " + "; ".join(last_errors))

    report["profile"] = profile(); report["user_id"] = user_id
    report["fixture_kind"] = (
        "historical_toolcoder" if profile() == "legacy"
        else "restbench_style_dev2026"
    )
    report["track_seed_strategy"] = (
        "artist_top_tracks" if profile() == "legacy"
        else "deterministic_named_track_searches"
    )
    return report
