#!/usr/bin/env bash
# Lifeboat Cache verification — one line:
#   curl -fsSL https://raw.githubusercontent.com/IterateAI/LifeboatCacheTest/main/run.sh | bash
# Options go after "bash -s --", e.g. ... | bash -s -- --yes --full
# Fetches lbc_verify.py from the same place and runs it with your engine's Python.
set -euo pipefail
REF=${LBC_VERIFY_REF:-main}
URL=https://raw.githubusercontent.com/IterateAI/LifeboatCacheTest/$REF/lbc_verify.py
DIR=${LBC_VERIFY_DIR:-$HOME/.lifeboatcache-verify}

case "$(uname -s)" in
  Linux) command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found: an NVIDIA driver is required" >&2; exit 1; } ;;
  Darwin) [ "$(uname -m)" = arm64 ] || { echo "macOS needs Apple silicon" >&2; exit 1; } ;;
  *) echo "Linux (NVIDIA GPU) or macOS (Apple silicon) is required" >&2; exit 1 ;;
esac
PY=$(command -v python3 || command -v python || true)
[ -n "$PY" ] || { echo "python3 not found" >&2; exit 1; }
mkdir -p "$DIR"
curl -fsSL "$URL" -o "$DIR/lbc_verify.py"
SUM=$( (sha256sum "$DIR/lbc_verify.py" 2>/dev/null || shasum -a 256 "$DIR/lbc_verify.py") | cut -c1-16)
echo "lbc_verify.py $SUM ($REF)"
# the verifier hands itself over to the Python that runs vLLM / SGLang once it has found it;
# keep the terminal attached for its one question (stopping a running server)
exec "$PY" "$DIR/lbc_verify.py" --workdir "$DIR" "$@"
