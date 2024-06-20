#!/bin/bash

set -e
cd "$(dirname "$0")"
cd manim-forge
maturin develop --release
cd ..
poetry run python "$@"
