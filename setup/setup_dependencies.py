#!/usr/bin/env python3
"""
Setup script for Beluga Challenge dependencies
"""

import subprocess
import sys

def install_package(package):
    """Install a package using pip"""
    try:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install {package}: {e}")
        return False

def main():
    """Main setup function"""
    print("=" * 60)
    print("BELUGA CHALLENGE - DEPENDENCY INSTALLATION")
    print("=" * 60)
    
    # Core dependencies
    dependencies = [
        "torch>=2.0.0",
        "numpy>=1.21.0", 
        "matplotlib>=3.5.0",
        "gymnasium>=0.29.0",
        "typing-extensions>=4.0.0",
        "pytest>=7.0.0",
        "jupyter>=1.0.0"
    ]
    
    # Optional dependencies
    optional_dependencies = [
        "pandas>=1.5.0",
        "seaborn>=0.11.0"
    ]
    
    # Upgrade pip first
    print("\nUpgrading pip...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        print("✓ pip upgraded successfully")
    except subprocess.CalledProcessError:
        print("⚠ Failed to upgrade pip, continuing anyway...")
    
    # Install core dependencies
    print(f"\nInstalling {len(dependencies)} core dependencies...")
    failed_packages = []
    
    for package in dependencies:
        if not install_package(package):
            failed_packages.append(package)
    
    # Install optional dependencies
    print(f"\nInstalling {len(optional_dependencies)} optional dependencies...")
    
    for package in optional_dependencies:
        if not install_package(package):
            print(f"⚠ Optional package {package} failed to install (not critical)")
    
    # Summary
    print("\n" + "=" * 60)
    print("INSTALLATION SUMMARY")
    print("=" * 60)
    
    if failed_packages:
        print(f"✗ {len(failed_packages)} core packages failed to install:")
        for package in failed_packages:
            print(f"  - {package}")
        print("\nPlease try installing these manually:")
        for package in failed_packages:
            print(f"  pip install {package}")
    else:
        print("✓ All core dependencies installed successfully!")
    
    print("\nYou can now run the Beluga Challenge:")
    print("  python -m rl.main --help")
    print("\nFor training:")
    print("  python -m rl.main --mode train")
    print("\nFor problem evaluation:")
    print("  python -m rl.main --mode problem --problem_path problems/problem_7_s49_j5_r2_oc85_f6.json")

if __name__ == "__main__":
    main()
