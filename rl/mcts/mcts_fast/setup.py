from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, Extension
import pybind11

# Define the extension module
ext_modules = [
    Pybind11Extension(
        "mcts_fast",
        [
            "bindings.cpp",
            "jig.cpp",
            "beluga.cpp",
            "rack.cpp",
            "production_line.cpp",
            "problem_state.cpp",
            "mcts_node.cpp",
            "mcts.cpp",
        ],
        include_dirs=[
            pybind11.get_cmake_dir() + "/../include",
            ".",
        ],
        language='c++',
        cxx_std=17,
    ),
]

setup(
    name="mcts_fast",
    version="0.1.0",
    author="Beluga Challenge Team",
    author_email="",
    description="Fast C++ implementation of MCTS for the Beluga Challenge",
    long_description="A high-performance C++ implementation of Monte Carlo Tree Search for the Beluga Challenge container optimization problem. Provides significant speed improvements over the Python implementation while maintaining full API compatibility.",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
    install_requires=[
        "pybind11>=2.6.0",
    ],
)