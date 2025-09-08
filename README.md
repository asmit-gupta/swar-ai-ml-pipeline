# 🎙️ Cladbe Audio AI

Advanced audio transcription and analysis system with speaker diarization and AI-powered insights for real estate sales calls.

## ✨ Features

- **Multi-format Audio Processing**: MP3, WAV, M4A, OGG, FLAC support
- **Speaker Diarization**: Open-source speaker separation (no API keys required)
- **Advanced Transcription**: Whisper-based transcription with parallel processing
- **AI Analysis**: Real estate sales call analysis via Ollama LLM
- **Professional Web Interface**: Responsive UI with multiple view modes

## 🚀 Quick Start

```bash
# Setup (one-time)
chmod +x scripts/*.sh
./scripts/setup.sh

# Run application
./scripts/run.sh

# Visit http://localhost:5000
```

## 📁 Project Structure

```
cladbe-audio-ai/
├── app.py                          # Flask web application
├── transcribe.py                   # Core transcription & diarization
├── llm_analysis.py                 # Ollama LLM integration
├── requirements.txt                # Python dependencies
├── templates/index.html            # Web interface
├── scripts/                        # Automation scripts
│   ├── setup.sh                   # Environment setup
│   ├── run.sh                     # Start application
│   └── stop.sh                    # Stop application
└── sample sales call amiltus.mp3  # Test audio file
```

## 🔧 Technology Stack

- **Backend**: Flask, OpenAI Whisper, SpeechBrain
- **Diarization**: SpeechBrain, Spectral Clustering, Simple VAD
- **AI Analysis**: Ollama (local LLM)
- **Frontend**: HTML5, CSS3, JavaScript
- **Audio Processing**: FFmpeg, PyTorch, librosa

## 📊 Processing Pipeline

1. **Audio Upload** → Web interface accepts audio files
2. **Transcription** → Whisper converts speech to text
3. **Diarization** → Open-source speaker separation
4. **Integration** → Timeline-based speaker assignment
5. **Analysis** → Ollama generates real estate insights
6. **Display** → Multi-view results interface

## 🎯 Usage

1. Upload audio file via web interface
2. Wait for processing (typically 30-60 seconds)
3. View results in multiple formats:
   - **AI Analysis**: Real estate sales insights
   - **Speaker View**: Conversation separated by speaker
   - **Raw Transcript**: Timestamped segments
   - **Speaker Stats**: Analytics and metrics

## 🛠️ Requirements

- Python 3.8+
- FFmpeg
- Ollama (for AI analysis)
- 4GB+ RAM recommended