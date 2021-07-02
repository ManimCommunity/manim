#!/usr/bin/env python
"""
This is intended to be run from the project root.
"""

import os

grpc_dir = "manim/grpc"
CMD_STRING = f"""
poetry run python \
    -m grpc_tools.protoc \
    -I {grpc_dir}/proto \
    --python_out={grpc_dir}/gen \
    --grpc_python_out={grpc_dir}/gen \
        {grpc_dir}/proto/frameserver.proto \
        {grpc_dir}/proto/renderserver.proto \
        {grpc_dir}/proto/threejs.proto
"""
os.system(CMD_STRING)
