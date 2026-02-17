cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(SIGA LANGUAGES CXX CUDA) # Include CUDA in project setup

set(CMAKE_VERBOSE_MAKEFILE ON)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17) # Set CUDA standard
set(CMAKE_CUDA_ARCHITECTURES 86)

set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G")

#set(CUDAToolkit_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9")
find_package(CUDAToolkit REQUIRED)
#find_package(cudss REQUIRED) # COMPONENTS cudss)

#set(CUDSS_DIR "C:/Program Files/NVIDIA cuDSS/v0.7" CACHE PATH "Path to cuDSS")

#if (MSVC)
#  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
#endif()

# Include directories
include_directories("${PROJECT_SOURCE_DIR}/src")
include_directories("${PROJECT_SOURCE_DIR}/external/eigen")
#include_directories("${CUDSS_DIR}/include")

add_subdirectory(src)
add_subdirectory(example)
