#!/bin/bash

# Factuality Evaluation Project Setup Script
# MSc AI Thesis Project - University of Manchester

echo "🚀 Setting up Factuality Evaluation Environment..."

# Check if Python 3.8+ is available
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3,8) else 1)" 2>/dev/null; then
    echo "❌ Python 3.8+ required. Current version: $python_version"
    exit 1
fi

echo "✅ Python version check passed"

# Create virtual environment
echo "📦 Creating virtual environment..."
if [ -d "venv" ]; then
  echo "⚠️  Virtual environment already exists. Removing and recreating..."
  rm -rf venv
fi
  python3 -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Download spaCy model for text processing
echo "🔤 Downloading spaCy model..."
python -m spacy download en_core_web_sm

# Install PyTorch with appropriate backend
echo "🔥 Installing PyTorch..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS - use MPS if available
    pip install torch torchvision torchaudio
    echo "✅ PyTorch installed with Metal Performance Shaders (MPS) support"
else
    # Linux - detect CUDA
    if command -v nvidia-smi &> /dev/null; then
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        echo "✅ PyTorch installed with CUDA support"
    else
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        echo "✅ PyTorch installed with CPU support"
    fi
fi

# Check GPU availability
echo "🖥️ Checking compute devices..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
if torch.cuda.is_available():
    print(f'✅ CUDA available - GPU: {torch.cuda.get_device_name(0)}')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('✅ MPS (Apple Silicon) available')
else:
    print('⚠️ CPU only - consider cloud GPU for large experiments')
"

echo "✅ Setup complete! To activate environment run: source venv/bin/activate"
echo "🎯 Next step: python src/data/download_datasets.py"
