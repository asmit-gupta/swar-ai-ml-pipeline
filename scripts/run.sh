#!/bin/bash

echo "🚀 Starting Cladbe Audio AI..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run ./setup.sh first"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check if ollama is running
echo "🔍 Checking Ollama status..."
if ! pgrep -f "ollama" > /dev/null; then
    echo "🟡 Starting Ollama service..."
    ollama serve &
    sleep 3
    
    # Wait for Ollama to be ready
    echo "⏳ Waiting for Ollama to initialize..."
    while ! curl -s http://localhost:11434/api/version > /dev/null 2>&1; do
        echo "   ... waiting for Ollama"
        sleep 2
    done
    echo "✅ Ollama is ready!"
else
    echo "✅ Ollama is already running"
fi

# Check if model is available
echo "🤖 Verifying nous-hermes2 model..."
if ! ollama list | grep -q "nous-hermes2"; then
    echo "📥 Downloading nous-hermes2 model (this may take a few minutes)..."
    ollama pull nous-hermes2
fi

# Create uploads directory if it doesn't exist
mkdir -p uploads

echo ""
echo "🎯 Starting Flask application..."
echo "🌐 Open your browser to: http://localhost:5000"
echo "⏹️  Press Ctrl+C to stop the application"
echo ""

# Start Flask app
python app.py