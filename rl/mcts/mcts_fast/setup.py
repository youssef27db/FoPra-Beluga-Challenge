from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, find_packages, Extension
import pybind11
import os
import sys
import platform

# Platform-specific compiler settings
if platform.system() == "Darwin":  # macOS
    os.environ.setdefault("CC", "clang")
    os.environ.setdefault("CXX", "clang++")
    # Disable AddressSanitizer explicitly on macOS
    os.environ["ASAN_OPTIONS"] = "detect_leaks=0:disable_coredump=0:unmap_shadow_on_exit=1"
elif platform.system() == "Windows":
    # Windows typically uses MSVC, which is automatically selected
    pass
else:  # Linux and other Unix-like systems
    os.environ.setdefault("CC", "gcc")
    os.environ.setdefault("CXX", "g++")
    # Disable AddressSanitizer explicitly on Linux
    os.environ["ASAN_OPTIONS"] = "detect_leaks=0:disable_coredump=0:unmap_shadow_on_exit=1"

# Define the extension module
ext_modules = [
    Pybind11Extension(
        "mcts_fast",  # Name it directly as mcts_fast
        sources=[
            "Jig.cpp",
            "Beluga.cpp", 
            "Rack.cpp",
            "ProductionLine.cpp",
            "ProblemState.cpp",
            "Action.cpp",
            "MCTSNode.cpp",
            "MCTS.cpp",
            "pybind_bindings.cpp",
        ],
        include_dirs=[
            pybind11.get_include(),
        ],
        cxx_std=17,
        # Platform-specific compiler and linker flags
        **({"extra_compile_args": ["/O2", "/DNDEBUG", "/EHsc"] if platform.system() == "Windows" 
            else {"extra_compile_args": [
                "-O3",  # Maximum optimization for speed
                "-DNDEBUG",  # Define NDEBUG to disable assertions
                "-fno-sanitize=address",  # Explicitly disable sanitizers
                "-fno-sanitize=undefined",
            ],
            "extra_link_args": [
                "-fno-sanitize=address",  # Explicitly disable sanitizers
                "-fno-sanitize=undefined",
            ]}
        }),
    ),
]

setup(
    name="mcts_fast",
    version="0.1.0",
    author="Nils-Frederik Schulze",
    description="A fast MCTS implementation with C++ backend and Python bindings",
    long_description="Monte Carlo Tree Search implementation with C++ backend for high performance and Python bindings via pybind11.",
    long_description_content_type="text/markdown",
    url="https://github.com/youssef27db/FoPra-Beluga-Challenge",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.19.0",  # Add other dependencies your module needs
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "Operating System :: OS Independent",
    ],
)
