#!/usr/bin/env bash
    
# move to project root directory
cd "$(dirname $0)/.." || { echo "Failed to change dicrectory to project root"; exit 1; }

if [ ! -d .venv ]; then
    echo "You need to create a virtual environment first."
    return 1
fi

source .venv/bin/activate || { echo "Failed to activate virtual environment"; exit 1; }

if [ -n "$SPECEDGE_OPTIMIZATION" ] && [ "$SPECEDGE_OPTIMIZATION" -ge 1 ]; then
   python -O src/script/client.py
else 
    python src/script/client.py
fi
