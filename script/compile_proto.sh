#!/usr/bin/env bash
    
# move to project root directory
cd "$(dirname $0)/.." || { echo "Failed to change dicrectory to project root"; exit 1; }

if [ ! -d .venv ]; then
    echo "You need to create a virtual environment first."
    return 1
fi

source .venv/bin/activate || { echo "Failed to activate virtual environment"; exit 1; }
    
python -m grpc_tools.protoc \
    --proto_path=. \
    --python_out=./src/specedge_grpc \
    --grpc_python_out=./src/specedge_grpc \
    --pyi_out=./src/specedge_grpc \
    specedge.proto
