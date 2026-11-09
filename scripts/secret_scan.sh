#!/usr/bin/env bash
# Scan the whole git history and the working tree for secrets. Run before the first push.
set -euo pipefail
cd "$(dirname "$0")/.."
gitleaks git --redact -v .
gitleaks dir --redact -v .
echo "secret scan clean"
