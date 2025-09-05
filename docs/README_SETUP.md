# 🎙️ Cladbe Audio AI - Quick Setup Guide

## 🚀 One-Command Setup & Run

### Prerequisites
- Python 3.8+ installed
- Ollama installed (`curl -fsSL https://ollama.ai/install.sh | sh`)
- FFmpeg installed (`brew install ffmpeg` on macOS)

### Setup (First Time Only)
```bash
chmod +x setup.sh run.sh stop.sh
./setup.sh
```

### Run the Application
```bash
./run.sh
```

### Stop the Application
```bash
./stop.sh
```

## 📱 Usage
1. Open browser to: http://localhost:5000
2. Upload an audio file (MP3, WAV, M4A, OGG, FLAC)
3. Wait for transcription and analysis
4. View results in AI Analysis or Raw Transcript tabs

## 🧪 Test with Sample
The project includes `sample sales call amiltus.mp3` for testing.

## 📁 What Gets Created
- `venv/` - Python virtual environment
- `uploads/` - Temporary audio storage (auto-cleaned)
- Downloaded Whisper models in `~/.cache/whisper/`

## 🐛 Troubleshooting
- **Port 5000 busy**: Change port in `app.py` line 133
- **Ollama connection**: Ensure Ollama is running on port 11434
- **Permission denied**: Run `chmod +x *.sh` to make scripts executable
- **FFmpeg error**: Install with `brew install ffmpeg` (macOS) or `sudo apt install ffmpeg` (Linux)

## 🔄 Scripts Overview
- `setup.sh` - One-time environment setup
- `run.sh` - Start the application  
- `stop.sh` - Stop and cleanup
- `requirements.txt` - Python dependencies