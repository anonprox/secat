#!/usr/bin/env python3
"""Install the API-Bank snapshot used by the upstream ToolCoder repository.

The benchmark implementation is intentionally not copied into SECAT.  This
bootstrap keeps API-Bank's original ToolManager, APIs, data, and correctness
predicates intact, then SECAT evaluates against them at runtime.
"""
from __future__ import annotations

import argparse
import json
import hashlib
import os
import re
from pathlib import Path
import shutil
import tempfile
import urllib.request
import zipfile

from benchmarks.apibank_runtime import discover_apibank_root


PINNED_COMMIT = "0d30ce85c62ede6a405f20f701a89172e4911e65"
DEFAULT_URL = f"https://github.com/dhx20150812/toolcoder/archive/{PINNED_COMMIT}.zip"


def _safe_extract(zf: zipfile.ZipFile, destination: Path) -> None:
    base = destination.resolve()
    for member in zf.infolist():
        target = (destination / member.filename).resolve()
        try:
            target.relative_to(base)
        except ValueError as exc:
            raise RuntimeError(f"Unsafe archive member: {member.filename}") from exc
    zf.extractall(destination)


def _download(url: str, path: Path) -> None:
    request = urllib.request.Request(
        url, headers={"User-Agent": "SECAT-APIBank-bootstrap/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, path.open("wb") as out:
        shutil.copyfileobj(response, out)


def install_apibank(*, destination=None, source_archive=None, url=None, expected_sha256=None) -> Path:
    """Stage a validated tree, rolling back the prior install if promotion fails."""
    project = Path(__file__).resolve().parent
    destination = Path(destination or project / "benchmarks" / "api_bank_vendor").expanduser().resolve()
    source_url = str(url or os.getenv("APIBANK_ARCHIVE_URL") or DEFAULT_URL)

    with tempfile.TemporaryDirectory(prefix="secat_apibank_") as tmp_name:
        tmp = Path(tmp_name)
        archive = tmp / "source.zip"
        if source_archive is not None:
            source_path = Path(source_archive).expanduser().resolve()
            if not source_path.is_file():
                raise FileNotFoundError(f"API-Bank archive not found: {source_path}")
            shutil.copy2(source_path, archive)
            provenance = {"source": "local_archive", "archive": str(source_path)}
        else:
            print(f"[APIBANK] Downloading ToolCoder API-Bank snapshot from {source_url}")
            _download(source_url, archive)
            provenance = {"source": "url", "url": source_url}

        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        provenance["archive_sha256"] = digest
        expected = str(expected_sha256 or os.getenv("APIBANK_ARCHIVE_SHA256") or "").strip().lower()
        if expected:
            if not re.fullmatch(r"[0-9a-f]{64}", expected):
                raise RuntimeError("API-Bank expected SHA-256 must be exactly 64 hexadecimal characters")
            if digest.lower() != expected:
                raise RuntimeError(
                    f"API-Bank archive SHA-256 mismatch: expected {expected}, got {digest}")
            provenance["expected_archive_sha256"] = expected

        extracted = tmp / "extracted"
        extracted.mkdir()
        with zipfile.ZipFile(archive) as zf:
            _safe_extract(zf, extracted)
        source_root = discover_apibank_root(extracted)

        staged = destination.with_name(destination.name + ".staging")
        if staged.exists():
            shutil.rmtree(staged)
        staged.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_root, staged)
        available_levels = ["level-1-given-desc"]
        if (staged / "lv1-lv2-samples" / "level-2-toolsearcher").is_dir():
            available_levels.append("level-2-toolsearcher")
        provenance.update({
            "installed_from_subdir": str(source_root.relative_to(extracted)),
            "levels": available_levels,
        })
        (staged / "SOURCE.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8")
        # Validate the complete staged tree before replacing any existing vendor.
        discover_apibank_root(staged)
        backup = None
        if destination.exists():
            backup = Path(tempfile.mkdtemp(prefix=destination.name + ".backup-", dir=destination.parent))
            backup.rmdir()
            os.replace(destination, backup)
        try:
            os.replace(staged, destination)
        except BaseException:
            if backup is not None:
                os.replace(backup, destination)
            raise
        else:
            if backup is not None:
                shutil.rmtree(backup)
    print(f"[APIBANK] Installed at {destination}")
    return destination


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Install API-Bank for SECAT")
    parser.add_argument("--destination", default=None)
    parser.add_argument("--archive", default=None,
                        help="Use a local ToolCoder/API-Bank zip instead of downloading")
    parser.add_argument("--url", default=None)
    parser.add_argument("--sha256", default=None,
                        help="Optional expected SHA-256 for the ToolCoder source archive")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    install_apibank(destination=args.destination, source_archive=args.archive, url=args.url,
                    expected_sha256=args.sha256)


if __name__ == "__main__":
    main()
