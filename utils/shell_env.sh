#!/usr/bin/env bash
# Load a dotenv file into the current shell WITHOUT overriding variables that
# were already supplied by the caller. This matches python-dotenv's
# load_dotenv(..., override=False) behavior used by config.py.
#
# Values are passed as NUL-delimited records so spaces, quotes, # characters,
# and newlines cannot be reinterpreted by the shell. Only valid shell variable
# names are accepted.
secat_load_dotenv_preserve() {
  local env_file="${1:-.env}"
  [[ -f "$env_file" ]] || return 0

  local key value
  while IFS= read -r -d '' key && IFS= read -r -d '' value; do
    # Preserve any value already supplied by the parent shell/process.
    if [[ "${!key+x}" == "x" ]]; then
      continue
    fi
    printf -v "$key" '%s' "$value"
    export "$key"
  done < <(python - "$env_file" <<'PY'
import re, sys
from dotenv import dotenv_values

path = sys.argv[1]
for key, value in dotenv_values(path).items():
    if value is None or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(key or "")):
        continue
    sys.stdout.buffer.write(str(key).encode("utf-8") + b"\0")
    sys.stdout.buffer.write(str(value).encode("utf-8") + b"\0")
PY
  )
}
