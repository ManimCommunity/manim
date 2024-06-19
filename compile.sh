#!/bin/bash

set -e
cd "$(dirname "$0")"
cd manimrust
maturin develop --release
cd ..
poetry run python test-render.py
