#!/bin/bash
set -euo pipefail

{test_commands}

cd /tests 
uv run parser.py
