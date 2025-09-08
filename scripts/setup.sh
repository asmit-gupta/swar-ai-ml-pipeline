#!/bin/bash
set -e  # Exit on any error

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

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        log_error "This script should not be run as root. Run as a regular user with sudo privileges."
        exit 1
    fi
}

# Detect operating system
detect_os() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        log_info "Detected macOS"
    elif [[ -f /etc/os-release ]]; then
        if grep -q "Ubuntu" /etc/os-release; then
            OS="ubuntu"
            local ubuntu_version=$(lsb_release -r -s)
            log_info "Detected Ubuntu $ubuntu_version"
        else
            OS="linux"
            log_warning "Detected Linux (not Ubuntu). Script optimized for Ubuntu/macOS."
            read -p "Continue anyway? (y/n): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
        fi
    else
        log_error "Unsupported operating system"
        exit 1
    fi
}

# Check for Homebrew on macOS
check_homebrew() {
    if [[ "$OS" == "macos" ]]; then
        if ! command -v brew &> /dev/null; then
            log_info "Installing Homebrew..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            
            # Add Homebrew to PATH for current session
            if [[ -f "/opt/homebrew/bin/brew" ]]; then
                eval "$(/opt/homebrew/bin/brew shellenv)"
            elif [[ -f "/usr/local/bin/brew" ]]; then
                eval "$(/usr/local/bin/brew shellenv)"
            fi
            
            log_success "Homebrew installed"
        else
            log_info "Homebrew already installed"
        fi
    fi
}

# Update system packages
update_system() {
    log_info "Updating system packages..."
    
    if [[ "$OS" == "macos" ]]; then
        brew update > /dev/null 2>&1
        log_success "Homebrew packages updated"
    else
        sudo apt-get update -qq
        log_success "System packages updated"
    fi
}

# Install system dependencies
install_system_deps() {
    log_info "Installing system dependencies..."
    
    if [[ "$OS" == "macos" ]]; then
        # macOS packages via Homebrew
        local packages=(
            "python@3.11"
            "ffmpeg"
            "portaudio"
            "libsndfile"
            "git"
            "wget"
        )
        
        for package in "${packages[@]}"; do
            if ! brew list "$package" &> /dev/null; then
                log_info "Installing $package..."
                brew install "$package" > /dev/null 2>&1
                log_success "Installed $package"
            else
                log_info "$package already installed"
            fi
        done
        
        # Ensure Python 3 is available
        if [[ ! -L "/usr/local/bin/python3" ]] && [[ ! -L "/opt/homebrew/bin/python3" ]]; then
            brew link python@3.11 > /dev/null 2>&1
        fi
        
    else
        # Ubuntu packages via apt
        local packages=(
            "python3"
            "python3-pip" 
            "python3-venv"
            "python3-dev"
            "build-essential"
            "pkg-config"
            "ffmpeg"
            "libavcodec-dev"
            "libavformat-dev"
            "libavutil-dev"
            "libswscale-dev"
            "libswresample-dev"
            "libavfilter-dev"
            "libasound2-dev"
            "portaudio19-dev"
            "libsndfile1-dev"
            "curl"
            "wget"
            "git"
            "unzip"
        )
        
        for package in "${packages[@]}"; do
            if ! dpkg -l | grep -q "^ii  $package "; then
                log_info "Installing $package..."
                sudo apt-get install -y "$package" > /dev/null 2>&1
                log_success "Installed $package"
            else
                log_info "$package already installed"
            fi
        done
    fi
}

# Check Python version
check_python() {
    log_info "Checking Python version..."
    
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed"
        exit 1
    fi
    
    local python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
    local major=$(echo $python_version | cut -d'.' -f1)
    local minor=$(echo $python_version | cut -d'.' -f2)
    
    if [[ $major -lt 3 ]] || [[ $major -eq 3 && $minor -lt 8 ]]; then
        log_error "Python 3.8+ is required. Found Python $python_version"
        exit 1
    fi
    
    log_success "Python $python_version detected"
}

# Install Ollama
install_ollama() {
    log_info "Installing Ollama..."
    
    if command -v ollama &> /dev/null; then
        log_info "Ollama already installed"
    else
        log_info "Downloading and installing Ollama..."
        curl -fsSL https://ollama.ai/install.sh | sh
        log_success "Ollama installed"
        
        # Add user to ollama group if it exists
        if getent group ollama > /dev/null 2>&1; then
            sudo usermod -aG ollama $USER
            log_info "Added user to ollama group"
        fi
    fi
}

# Setup Python environment
setup_python_env() {
    log_info "Setting up Python virtual environment..."
    
    # Remove existing venv if it exists
    if [ -d "venv" ]; then
        log_warning "Removing existing virtual environment..."
        rm -rf venv
    fi
    
    # Create new virtual environment
    python3 -m venv venv
    log_success "Virtual environment created"
    
    # Activate virtual environment
    source venv/bin/activate
    log_success "Virtual environment activated"
    
    # Upgrade pip
    log_info "Upgrading pip..."
    pip install --upgrade pip > /dev/null 2>&1
    log_success "Pip upgraded"
}

# Install Python dependencies
install_python_deps() {
    log_info "Installing Python dependencies..."
    log_warning "This may take 10-15 minutes depending on your internet connection..."
    
    # Ensure we're in the virtual environment
    source venv/bin/activate
    
    # Install PyTorch first (CPU version)
    log_info "Installing PyTorch (CPU version)..."
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu > /dev/null 2>&1
    log_success "PyTorch installed"
    
    # Install other dependencies
    local deps=(
        "openai-whisper==20231117"
        "flask==3.0.0" 
        "requests==2.31.0"
        "numpy==1.24.3"
        "scipy==1.11.3"
        "librosa==0.10.1"
        "ffmpeg-python==0.2.0"
        "speechbrain==0.5.16"
        "scikit-learn==1.3.0"
    )
    
    for dep in "${deps[@]}"; do
        log_info "Installing $dep..."
        pip install "$dep" > /dev/null 2>&1
        log_success "Installed $dep"
    done
    
    log_success "All Python dependencies installed"
}

# Setup Ollama service and model
setup_ollama() {
    log_info "Setting up Ollama service..."
    
    # Start Ollama service
    if ! pgrep -f "ollama" > /dev/null; then
        log_info "Starting Ollama service..."
        ollama serve > /dev/null 2>&1 &
        sleep 5
        log_success "Ollama service started"
    else
        log_info "Ollama service already running"
    fi
    
    # Wait for Ollama to be ready
    log_info "Waiting for Ollama to initialize..."
    local attempts=0
    while ! curl -s http://localhost:11434/api/version > /dev/null 2>&1; do
        sleep 2
        attempts=$((attempts + 1))
        if [ $attempts -gt 30 ]; then
            log_error "Ollama service failed to start"
            exit 1
        fi
        echo -n "."
    done
    echo
    log_success "Ollama service is ready"
    
    # Download model
    log_info "Checking for nous-hermes2 model..."
    if ! ollama list | grep -q "nous-hermes2"; then
        log_info "Downloading nous-hermes2 model (this may take several minutes)..."
        ollama pull nous-hermes2
        log_success "nous-hermes2 model downloaded"
    else
        log_success "nous-hermes2 model already available"
    fi
}

# Create necessary directories
create_directories() {
    log_info "Creating necessary directories..."
    
    mkdir -p uploads
    mkdir -p tmp_vad
    mkdir -p tmp_spkrec
    
    log_success "Directories created"
}

# Verify installation
verify_installation() {
    log_info "Verifying installation..."
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Test imports
    local test_imports=(
        "import flask"
        "import whisper" 
        "import torch"
        "import numpy"
        "import librosa"
        "import speechbrain"
        "import sklearn"
    )
    
    for import_test in "${test_imports[@]}"; do
        if python3 -c "$import_test" 2>/dev/null; then
            log_success "$(echo $import_test | cut -d' ' -f2) import successful"
        else
            log_error "$(echo $import_test | cut -d' ' -f2) import failed"
            return 1
        fi
    done
    
    # Test Ollama
    if curl -s http://localhost:11434/api/version > /dev/null 2>&1; then
        log_success "Ollama service is running"
    else
        log_warning "Ollama service not responding"
    fi
    
    log_success "Installation verification complete"
}

# Main installation function
main() {
    echo "🚀 CLADBE AUDIO AI - CROSS-PLATFORM SETUP"
    echo "==========================================="
    echo "This script will install all dependencies for Cladbe Audio AI"
    echo "Estimated time: 15-20 minutes"
    echo ""
    
    check_root
    detect_os
    
    if [[ "$OS" == "macos" ]]; then
        check_homebrew
    fi
    
    log_info "Starting installation process..."
    
    update_system
    install_system_deps
    check_python
    install_ollama
    setup_python_env
    install_python_deps
    setup_ollama
    create_directories
    verify_installation
    
    echo ""
    echo "🎉 INSTALLATION COMPLETE!"
    echo "========================"
    log_success "Cladbe Audio AI is now ready to use!"
    echo ""
    echo "📋 Next steps:"
    echo "   1. Run: ./scripts/run.sh"
    echo "   2. Open: http://localhost:5000"
    echo "   3. Upload an audio file to test"
    echo ""
    echo "📊 System Status:"
    echo "   ✅ Python environment: Ready"
    echo "   ✅ Audio processing: Ready" 
    echo "   ✅ Speaker diarization: Ready"
    echo "   ✅ AI analysis: Ready"
    echo ""
    echo "💡 Troubleshooting:"
    echo "   - If you get permission errors, make sure you're not running as root"
    echo "   - If Ollama fails, try: sudo systemctl restart ollama"
    echo "   - For memory issues, ensure you have 8GB+ RAM available"
    echo ""
    echo "🎯 Ready to process your first audio file!"
}

# Handle script interruption
trap 'log_error "Installation interrupted!"; exit 1' INT TERM

# Run main function
main "$@"