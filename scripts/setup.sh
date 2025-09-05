#!/bin/bash

echo "🚀 Setting up Cladbe Audio AI..."

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📥 Installing Python dependencies..."
pip install openai-whisper torch torchaudio flask requests numpy scipy librosa ffmpeg-python

# Check if ollama is running
echo "🔍 Checking Ollama status..."
if ! pgrep -f "ollama" > /dev/null; then
    echo "⚠️  Ollama not running. Starting Ollama in background..."
    ollama serve &
    sleep 5
fi

# Pull required model if not exists
echo "🤖 Checking for nous-hermes2 model..."
if ! ollama list | grep -q "nous-hermes2"; then
    echo "📥 Downloading nous-hermes2 model..."
    ollama pull nous-hermes2
else
    echo "✅ nous-hermes2 model already available"
fi

# Create uploads directory
echo "📁 Creating uploads directory..."
mkdir -p uploads

echo "✅ Setup complete!"
echo ""
echo "🎯 To run the application:"
echo "   ./run.sh"
echo ""
echo "🌐 Access the app at: http://localhost:5000"