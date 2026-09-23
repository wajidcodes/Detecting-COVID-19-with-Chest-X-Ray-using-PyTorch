#!/bin/bash
echo "🚀 Starting PulmoScan AI..."
echo "🌐 The UI will be available at: http://127.0.0.1:8000"
echo ""

# Activate virtual environment if available
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run FastAPI server directly
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
