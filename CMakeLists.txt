cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(SIGA LANGUAGES CXX CUDA) # Include CUDA in project setup

set(CMAKE_VERBOSE_MAKEFILE ON)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17) # Set CUDA standard
set(CMAKE_CUDA_ARCHITECTURES 75)

set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G")

# Include directories
include_directories("${PROJECT_SOURCE_DIR}/src")
include_directories("${PROJECT_SOURCE_DIR}/external/eigen")

add_subdirectory(src)
add_subdirectory(example)
