#!/bin/bash

# Beluga Challenge Dependencies Installation Script for Ubuntu/Linux
# Author: Beluga Challenge Team
# Date: $(date +%Y-%m-%d)

echo "============================================================"
echo "BELUGA CHALLENGE - DEPENDENCY INSTALLATION (Ubuntu/Linux)"
echo "============================================================"
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check if Python 3 is installed
print_step "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3 first:"
    echo "  sudo apt update"
    echo "  sudo apt install python3 python3-pip python3-venv"
    exit 1
else
    PYTHON_VERSION=$(python3 --version)
    print_status "Found: $PYTHON_VERSION"
fi

# Check if pip is installed
print_step "Checking pip installation..."
if ! command -v pip3 &> /dev/null; then
    print_warning "pip3 not found. Installing pip..."
    sudo apt update
    sudo apt install python3-pip -y
else
    print_status "pip3 is available"
fi

# Create virtual environment (recommended)
print_step "Do you want to create a virtual environment? (recommended) [Y/n]"
read -r create_venv
if [[ $create_venv != "n" && $create_venv != "N" ]]; then
    print_step "Creating virtual environment..."
    python3 -m venv beluga_env
    print_status "Virtual environment created: beluga_env"
    print_status "Activating virtual environment..."
    source beluga_env/bin/activate
    print_status "Virtual environment activated"
    echo
    print_warning "Remember to activate the environment in future sessions:"
    echo "  source beluga_env/bin/activate"
    echo
fi

# Upgrade pip
print_step "Upgrading pip..."
python3 -m pip install --upgrade pip
if [ $? -eq 0 ]; then
    print_status "pip upgraded successfully"
else
    print_warning "Failed to upgrade pip, continuing anyway..."
fi

# Install system dependencies that might be needed
print_step "Installing system dependencies..."
sudo apt update
sudo apt install -y python3-dev build-essential libssl-dev libffi-dev

# Install core Python dependencies
print_step "Installing core Python dependencies..."

dependencies=(
    "torch>=2.0.0"
    "numpy>=1.21.0"
    "matplotlib>=3.5.0"
    "gymnasium>=0.29.0"
    "typing-extensions>=4.0.0"
    "pytest>=7.0.0"
    "jupyter>=1.0.0"
)

failed_packages=()

for package in "${dependencies[@]}"; do
    print_step "Installing $package..."
    python3 -m pip install "$package"
    if [ $? -eq 0 ]; then
        print_status "✓ Successfully installed $package"
    else
        print_error "✗ Failed to install $package"
        failed_packages+=("$package")
    fi
done

# Install optional dependencies
print_step "Installing optional dependencies..."
optional_deps=(
    "pandas>=1.5.0"
    "seaborn>=0.11.0"
)

for package in "${optional_deps[@]}"; do
    print_step "Installing optional package: $package..."
    python3 -m pip install "$package"
    if [ $? -eq 0 ]; then
        print_status "✓ Successfully installed $package"
    else
        print_warning "⚠ Optional package $package failed to install (not critical)"
    fi
done

# Summary
echo
echo "============================================================"
echo "INSTALLATION SUMMARY"
echo "============================================================"

if [ ${#failed_packages[@]} -eq 0 ]; then
    print_status "✓ All core dependencies installed successfully!"
else
    print_error "✗ ${#failed_packages[@]} core packages failed to install:"
    for package in "${failed_packages[@]}"; do
        echo "  - $package"
    done
    echo
    print_warning "Please try installing these manually:"
    for package in "${failed_packages[@]}"; do
        echo "  python3 -m pip install \"$package\""
    done
fi

echo
print_status "You can now run the Beluga Challenge:"
echo "  python3 -m rl.main --help"
echo
print_status "For training:"
echo "  python3 -m rl.main --mode train"
echo
print_status "For problem evaluation:"
echo "  python3 -m rl.main --mode problem --problem_path problems/problem_7_s49_j5_r2_oc85_f6.json"

if [[ $create_venv != "n" && $create_venv != "N" ]]; then
    echo
    print_warning "Don't forget to activate your virtual environment in future sessions:"
    echo "  source beluga_env/bin/activate"
fi

echo
print_status "Installation completed!"
