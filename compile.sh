#!/bin/bash

cd manimrust || true
maturin develop || exit 1
cd .. || true
python test-render.py
