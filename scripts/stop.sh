#!/bin/bash

echo "🛑 Stopping Cladbe Audio AI..."

# Kill Flask process
echo "🔸 Stopping Flask application..."
pkill -f "python app.py" 2>/dev/null || echo "   Flask app not running"

# Optionally stop Ollama (uncomment if you want to stop Ollama too)
# echo "🔸 Stopping Ollama service..."
# pkill -f "ollama serve" 2>/dev/null || echo "   Ollama not running"

# Clean up any temporary files
echo "🧹 Cleaning up temporary files..."
rm -rf uploads/*.mp3 uploads/*.wav 2>/dev/null || true

# Deactivate virtual environment if active
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "🔧 Deactivating virtual environment..."
    deactivate 2>/dev/null || true
fi

echo "✅ Cleanup complete!"