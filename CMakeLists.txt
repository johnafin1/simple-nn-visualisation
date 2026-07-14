cmake_minimum_required(VERSION 3.20)

project(simple_nn_visualisation
    VERSION 0.0.1
    DESCRIPTION "A neural network built from scratch in C++20 with visualisation"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type" FORCE)
endif()

# Shared warning flags as an interface target so every target opts in the same way.
add_library(nn_warnings INTERFACE)
target_compile_options(nn_warnings INTERFACE
    -Wall -Wextra -Wpedantic)

# Public headers (declarations) live under src/includes and are on the include path.
# Code includes them as e.g. #include "helpers/linalg.hpp".
add_library(nn_headers INTERFACE)
target_include_directories(nn_headers INTERFACE ${CMAKE_SOURCE_DIR}/src/includes)

enable_testing()

add_subdirectory(src)
add_subdirectory(tests)
