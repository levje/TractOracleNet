# Install required packages
# Print python version
echo "Python version: $(python --version)"
echo "Installing required packages"
pip install Cython numpy packaging --quiet
# If platform has CUDA installed
if [ -x "$(command -v nvidia-smi)" ]; then
    # Print GPU name
    echo "Found GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    # Print CUDA version from grepping nvidia-smi
    echo "Found CUDA version: $(nvidia-smi | grep "CUDA Version" | awk '{print $9}')"
    # Check CUDA version and format as cuXXX
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | sed 's/\.//g')
    if (( $CUDA_VERSION == 116 )); then
        CUDA_VERSION="cu116"
    elif (( $CUDA_VERSION >= 117 )); then
        CUDA_VERSION="cu117"
    else
      CUDA_VERSION="cpu"
      echo "No suitable CUDA version found. Installing PyTorch without CUDA support"
    fi
else
    echo "No GPU or CUDA installation found. Installing PyTorch without CUDA support"
    CUDA_VERSION="cpu"
fi

if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Platform: macOS"
    echo "Installing Pytorch 1.13.1"
    pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --quiet
else
    # Install pytorch
    echo "Installing Pytorch 1.13.1+${CUDA_VERSION}"
    # Install PyTorch with CUDA support
    pip install torch==1.13.1+${CUDA_VERSION} torchvision==0.14.1+${CUDA_VERSION} torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/${CUDA_VERSION} --quiet
fi

# Install other required packages and modules
echo "Installing rest of packages"
pip install -e . --quiet
