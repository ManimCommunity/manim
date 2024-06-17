#!/bin/bash

cd manimrust || true
maturin develop --release || exit 1
cd .. || true
python test-render.py
