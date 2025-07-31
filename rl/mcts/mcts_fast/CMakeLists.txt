cmake_minimum_required(VERSION 3.12)
project(mcts_fast)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Special handling for macOS with Homebrew-installed OpenMP
if(APPLE)
    # Find Homebrew's libomp installation
    execute_process(
        COMMAND brew --prefix libomp
        OUTPUT_VARIABLE HOMEBREW_LIBOMP_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    
    if(HOMEBREW_LIBOMP_PREFIX)
        # Set up include and library paths for Homebrew's OpenMP
        include_directories(${HOMEBREW_LIBOMP_PREFIX}/include)
        link_directories(${HOMEBREW_LIBOMP_PREFIX}/lib)
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY ${HOMEBREW_LIBOMP_PREFIX}/lib/libomp.dylib)
        
        # Force-set OpenMP_CXX_FOUND to TRUE
        set(OpenMP_CXX_FOUND TRUE)
        
        message(STATUS "Using Homebrew's OpenMP installation at ${HOMEBREW_LIBOMP_PREFIX}")
        add_compile_definitions(HAS_OPENMP=1)
    else()
        message(STATUS "OpenMP not found on macOS. Using fallback mode.")
        add_compile_definitions(HAS_OPENMP=0)
    endif()
else()
    # For other platforms, look for OpenMP normally
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        add_compile_definitions(HAS_OPENMP=1)
        message(STATUS "OpenMP found. Parallel execution will be enabled.")
    else()
        add_compile_definitions(HAS_OPENMP=0)
        message(STATUS "OpenMP not found. Falling back to single-threaded execution.")
    endif()
endif()

# Add the source files for our core library
set(SOURCES
    Jig.cpp
    Beluga.cpp
    Rack.cpp
    ProductionLine.cpp
    ProblemState.cpp
    Action.cpp
    MCTSNode.cpp
    MCTS.cpp
)

# Create a standalone test executable for testing problem loading
add_executable(test_cpp_only
    ${SOURCES}
    test_cpp_only.cpp
)
target_compile_features(test_cpp_only PRIVATE cxx_std_17)

# Create a standalone test executable for full MCTS
add_executable(test_mcts
    ${SOURCES}
    test_mcts.cpp
)
target_compile_features(test_mcts PRIVATE cxx_std_17)

# Optional: Add compiler flags for optimization
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(test_cpp_only PRIVATE -O3)
    target_compile_options(test_mcts PRIVATE -O3)
endif()

# Include directories
target_include_directories(test_cpp_only PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(test_mcts PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Create thread benchmark executable
add_executable(benchmark_threads
    ${SOURCES}
    benchmark_threads.cpp
)
target_compile_features(benchmark_threads PRIVATE cxx_std_17)
target_include_directories(benchmark_threads PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Create a simple test executable to debug issues
add_executable(simple_test
    ${SOURCES}
    simple_test.cpp
)
target_compile_features(simple_test PRIVATE cxx_std_17)
target_include_directories(simple_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Create a simple benchmark executable
add_executable(simple_benchmark
    ${SOURCES}
    simple_benchmark.cpp
)
target_compile_features(simple_benchmark PRIVATE cxx_std_17)
target_include_directories(simple_benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Create a thread comparison executable
add_executable(thread_comparison
    ${SOURCES}
    thread_comparison.cpp
)
target_compile_features(thread_comparison PRIVATE cxx_std_17)
target_include_directories(thread_comparison PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Create a minimal test executable for debug
add_executable(minimal_test
    ${SOURCES}
    minimal_test.cpp
)
target_compile_features(minimal_test PRIVATE cxx_std_17)
target_include_directories(minimal_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Create a thread benchmark executable
add_executable(thread_benchmark
    ${SOURCES}
    thread_benchmark.cpp
)
target_compile_features(thread_benchmark PRIVATE cxx_std_17)
target_include_directories(thread_benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Optional: Add compiler flags for optimization
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(benchmark_threads PRIVATE -O3)
    target_compile_options(simple_test PRIVATE -O3)
    target_compile_options(thread_benchmark PRIVATE -O3)
endif()

# Link OpenMP if found
if(OpenMP_CXX_FOUND)
    if(APPLE AND DEFINED HOMEBREW_LIBOMP_PREFIX)
        # On macOS with Homebrew OpenMP, we need to link explicitly
        target_compile_options(test_cpp_only PRIVATE -Xpreprocessor -fopenmp)
        target_link_libraries(test_cpp_only PRIVATE ${OpenMP_omp_LIBRARY})
        
        target_compile_options(test_mcts PRIVATE -Xpreprocessor -fopenmp)
        target_link_libraries(test_mcts PRIVATE ${OpenMP_omp_LIBRARY})
        
        target_compile_options(benchmark_threads PRIVATE -Xpreprocessor -fopenmp)
        target_link_libraries(benchmark_threads PRIVATE ${OpenMP_omp_LIBRARY})
        
        target_compile_options(simple_test PRIVATE -Xpreprocessor -fopenmp)
        target_link_libraries(simple_test PRIVATE ${OpenMP_omp_LIBRARY})
        
        target_compile_options(simple_benchmark PRIVATE -Xpreprocessor -fopenmp)
        target_link_libraries(simple_benchmark PRIVATE ${OpenMP_omp_LIBRARY})
        
        target_compile_options(thread_comparison PRIVATE -Xpreprocessor -fopenmp)
        target_link_libraries(thread_comparison PRIVATE ${OpenMP_omp_LIBRARY})
        
        target_compile_options(minimal_test PRIVATE -Xpreprocessor -fopenmp)
        target_link_libraries(minimal_test PRIVATE ${OpenMP_omp_LIBRARY})
        
        target_compile_options(thread_benchmark PRIVATE -Xpreprocessor -fopenmp)
        target_link_libraries(thread_benchmark PRIVATE ${OpenMP_omp_LIBRARY})
    else()
        # Standard OpenMP linking
        target_link_libraries(test_cpp_only PRIVATE OpenMP::OpenMP_CXX)
        target_link_libraries(test_mcts PRIVATE OpenMP::OpenMP_CXX)
        target_link_libraries(benchmark_threads PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()

# Note: pybind11 module is disabled for now to focus on C++ functionality
# To re-enable it later, we can add:
# include(FetchContent)
# FetchContent_Declare(pybind11 GIT_REPOSITORY https://github.com/pybind/pybind11.git GIT_TAG v2.11.1)
# FetchContent_MakeAvailable(pybind11)
# pybind11_add_module(mcts_fast ${SOURCES} pybind_bindings.cpp)
