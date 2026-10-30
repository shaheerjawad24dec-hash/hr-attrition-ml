#!/usr/bin/env bash
# Install every library, train the models and run the tests on macOS/Linux.
# Run: bash setup_mac_linux.sh
set -e
cd "$(dirname "$0")"
for candidate in python3.13 python3.12 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
        exec "$candidate" setup_project.py
    fi
done
echo "Python 3.12+ not found. Install it from https://www.python.org/downloads/"
exit 1
