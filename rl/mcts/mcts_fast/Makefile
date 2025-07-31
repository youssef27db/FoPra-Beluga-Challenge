# Makefile für MCTS pybind-test

.PHONY: all build install clean test test-complete cmake-build

# Default target
all: build

# Build with pip
build:
	pip install .

# Install in development mode
install-dev:
	pip install -e .

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -name "*.so" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +

# Run basic tests
test: build
	python test_bindings.py

# Run complete MCTS tests
test-complete: build
	python test_mcts_complete.py

# Run all tests
test-all: test test-complete

# CMake build (alternative)
cmake-build:
	mkdir -p build
	cd build && cmake ..
	cd build && make -j4

# Clean CMake build
cmake-clean:
	rm -rf build/
	
# Shared compilation flags
CXX_FLAGS=-std=c++17 -O3

# Object files
MCTS.o: MCTS.cpp MCTS.hpp MCTSNode.hpp ProblemState.hpp
	g++ $(CXX_FLAGS) -c MCTS.cpp

MCTSNode.o: MCTSNode.cpp MCTSNode.hpp ProblemState.hpp
	g++ $(CXX_FLAGS) -c MCTSNode.cpp

ProblemState.o: ProblemState.cpp ProblemState.hpp
	g++ $(CXX_FLAGS) -c ProblemState.cpp

Action.o: Action.cpp Action.hpp
	g++ $(CXX_FLAGS) -c Action.cpp

Jig.o: Jig.cpp Jig.hpp
	g++ $(CXX_FLAGS) -c Jig.cpp

Beluga.o: Beluga.cpp Beluga.hpp
	g++ $(CXX_FLAGS) -c Beluga.cpp

Rack.o: Rack.cpp Rack.hpp
	g++ $(CXX_FLAGS) -c Rack.cpp

ProductionLine.o: ProductionLine.cpp ProductionLine.hpp
	g++ $(CXX_FLAGS) -c ProductionLine.cpp

# Compile test_root_parallel
test-parallel: test_root_parallel.o MCTS.o MCTSNode.o ProblemState.o Action.o Jig.o Beluga.o Rack.o ProductionLine.o
	g++ $(CXX_FLAGS) -pthread -o test_root_parallel test_root_parallel.o MCTS.o MCTSNode.o ProblemState.o Action.o Jig.o Beluga.o Rack.o ProductionLine.o

test_root_parallel.o: test_root_parallel.cpp MCTS.hpp MCTSNode.hpp ProblemState.hpp
	g++ $(CXX_FLAGS) -c test_root_parallel.cpp

# Install pybind11 if not present
install-deps:
	pip install pybind11

# Full setup from scratch
setup: install-deps build test-complete

# Build and run the thread benchmark
benchmark:
	mkdir -p build
	cd build && cmake -DCMAKE_BUILD_TYPE=Release ..
	cd build && make benchmark_threads -j4
	./build/benchmark_threads

# Run root parallelization test
run-parallel-test: test-parallel
	./test_root_parallel

# Help
help:
	@echo "Available targets:"
	@echo "  build          - Build with pip"
	@echo "  install-dev    - Install in development mode"
	@echo "  clean          - Clean build artifacts"
	@echo "  test           - Run basic tests"
	@echo "  test-complete  - Run complete MCTS tests"
	@echo "  test-all       - Run all tests"
	@echo "  test-parallel  - Compile root parallelization test"
	@echo "  run-parallel-test - Run root parallelization test"
	@echo "  cmake-build    - Build with CMake"
	@echo "  cmake-clean    - Clean CMake build"
	@echo "  benchmark      - Build and run thread benchmark"
	@echo "  install-deps   - Install pybind11"
	@echo "  setup          - Full setup from scratch"
	@echo "  help           - Show this help"
