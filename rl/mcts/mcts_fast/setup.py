from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, find_packages, Extension
import pybind11
import os
import sys

# Configure for production build without sanitizers
os.environ["CC"] = "clang"
os.environ["CXX"] = "clang++"

# Disable AddressSanitizer explicitly
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
        # Pure production flags with optimizations
        extra_compile_args=[
            "-O3",  # Maximum optimization for speed
            "-DNDEBUG",  # Define NDEBUG to disable assertions
            "-fno-sanitize=address",  # Explicitly disable sanitizers
            "-fno-sanitize=undefined",
        ],
        extra_link_args=[
            "-fno-sanitize=address",  # Explicitly disable sanitizers
            "-fno-sanitize=undefined",
        ],
    ),
]

setup(
    name="mcts_fast",
    version="0.1.0",
    author="Nils-Frederik Schulze",
    description="A fast MCTS implementation with C++ backend and Python bindings",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.6",
)
