#!/bin/bash

# Install script for mcts_fast
echo "Installing mcts_fast module..."

# Check Python version
python_version=$(python --version 2>&1)
echo "Using $python_version"

# Check operating system
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS specific setup
    echo "Detected macOS operating system"
    
    # Check if Homebrew is installed
    if ! command -v brew &> /dev/null; then
        echo "Homebrew not found. OpenMP support requires Homebrew and libomp."
        echo "Please install Homebrew from https://brew.sh/ and run 'brew install libomp'"
        echo "Then run this script again."
        
        read -p "Do you want to continue without OpenMP support? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Installation aborted."
            exit 1
        fi
    else
        # Check if libomp is installed
        if ! brew list --formula | grep -q "^libomp$"; then
            echo "Installing OpenMP support via Homebrew..."
            brew install libomp
        else
            echo "OpenMP support is already installed."
        fi
    fi
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux specific setup
    echo "Detected Linux operating system"
    
    # Check if we can use apt-get (Debian/Ubuntu)
    if command -v apt-get &> /dev/null; then
        echo "Checking OpenMP development libraries..."
        if ! dpkg -l | grep -q "libomp-dev"; then
            echo "Installing OpenMP development libraries..."
            sudo apt-get update
            sudo apt-get install -y libomp-dev
        else
            echo "OpenMP development libraries are already installed."
        fi
    fi
fi

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip setuptools wheel numpy pybind11

# Install the module
echo "Building and installing mcts_fast..."
pip install ./rl/mcts/mcts_fast

# Check if installation was successful
if [ $? -eq 0 ]; then
    echo "Installation successful!"
    echo "You can now import mcts_fast in your Python code."
else
    echo "Installation failed. Please check the error messages above."
    exit 1
fi
