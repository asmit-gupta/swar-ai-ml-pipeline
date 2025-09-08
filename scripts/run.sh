#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

echo "🚀 STARTING CLADBE AUDIO AI"
echo "============================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    log_error "Virtual environment not found"
    echo ""
    echo "Please run the setup first:"
    echo "  chmod +x scripts/setup.sh"
    echo "  ./scripts/setup.sh"
    exit 1
fi

# Activate virtual environment
log_info "Activating virtual environment..."
source venv/bin/activate
log_success "Virtual environment activated"

# Check if Flask app exists
if [ ! -f "app.py" ]; then
    log_error "app.py not found. Make sure you're in the correct directory."
    exit 1
fi

# Check if ollama is installed
if ! command -v ollama &> /dev/null; then
    log_warning "Ollama not found. AI analysis will not work."
    echo "To install Ollama: curl -fsSL https://ollama.ai/install.sh | sh"
else
    # Check if ollama is running
    log_info "Checking Ollama service..."
    if ! pgrep -f "ollama" > /dev/null; then
        log_info "Starting Ollama service..."
        ollama serve > /dev/null 2>&1 &
        sleep 3
        
        # Wait for Ollama to be ready with timeout
        log_info "Waiting for Ollama to initialize..."
        attempts=0
        while ! curl -s http://localhost:11434/api/version > /dev/null 2>&1; do
            sleep 2
            attempts=$((attempts + 1))
            if [ $attempts -gt 15 ]; then
                log_warning "Ollama service taking too long to start. Continuing without AI analysis."
                break
            fi
            echo -n "."
        done
        
        if [ $attempts -le 15 ]; then
            echo
            log_success "Ollama service is ready"
            
            # Check if model is available
            log_info "Verifying nous-hermes2 model..."
            if ! ollama list | grep -q "nous-hermes2" 2>/dev/null; then
                log_warning "nous-hermes2 model not found"
                echo "AI analysis will be limited. To download:"
                echo "  ollama pull nous-hermes2"
            else
                log_success "nous-hermes2 model ready"
            fi
        fi
    else
        log_success "Ollama service already running"
        
        # Verify model
        if ollama list | grep -q "nous-hermes2" 2>/dev/null; then
            log_success "nous-hermes2 model ready"
        else
            log_warning "nous-hermes2 model not found"
        fi
    fi
fi

# Create necessary directories
log_info "Preparing directories..."
mkdir -p uploads
mkdir -p tmp_vad  
mkdir -p tmp_spkrec
log_success "Directories ready"

# Check system resources
log_info "Checking system resources..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    total_mem=$(sysctl -n hw.memsize | awk '{printf "%.0f", $1/1024/1024}')
else
    # Linux
    total_mem=$(free -m | awk 'NR==2{printf "%.0f", $2}')
fi

if [ "$total_mem" -lt 4096 ]; then
    log_warning "Low memory detected (${total_mem}MB). Recommended: 8GB+ for optimal performance."
fi

# Start Flask application
echo ""
echo "🎯 STARTING APPLICATION"
echo "========================"
log_success "All systems ready!"
echo ""
echo "🌐 Application will be available at: http://localhost:5000"
echo "📱 For remote access, use: http://YOUR_SERVER_IP:5000"
echo "⏹️  Press Ctrl+C to stop the application"
echo ""

# Handle graceful shutdown
cleanup() {
    echo ""
    log_info "Shutting down application..."
    
    # Kill any background processes
    jobs -p | xargs -r kill 2>/dev/null
    
    log_success "Application stopped"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start Flask app with better error handling
log_info "Starting Flask server..."
if ! python app.py; then
    echo ""
    log_error "Flask application failed to start"
    echo ""
    echo "🔧 Troubleshooting:"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "   - Check if port 5000 is already in use: lsof -i :5000"
    else
        echo "   - Check if port 5000 is already in use: sudo netstat -tlnp | grep :5000"
    fi
    echo "   - Verify virtual environment: which python"
    echo "   - Check dependencies: pip list"
    echo "   - View detailed error logs above"
    exit 1
fi