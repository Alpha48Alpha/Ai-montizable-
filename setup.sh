#!/usr/bin/env bash
# setup.sh — Bootstrap the Alpha48Alpha AI Lab Python environment
#
# Usage:
#   bash setup.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="alpha48alpha_ai_lab"

mkdir -p "$SCRIPT_DIR/$PROJECT_DIR"
cd "$SCRIPT_DIR/$PROJECT_DIR"

python -m venv venv
# shellcheck disable=SC1091
source venv/bin/activate

pip install --upgrade pip
pip install -r "$SCRIPT_DIR/requirements.txt"

echo ""
echo "✅ Environment ready. Activate it with:"
echo "   source $SCRIPT_DIR/$PROJECT_DIR/venv/bin/activate"
