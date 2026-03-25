cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(SIGA LANGUAGES CXX CUDA) # Include CUDA in project setup

set(CMAKE_VERBOSE_MAKEFILE ON)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17) # Set CUDA standard
set(CMAKE_CUDA_ARCHITECTURES 80 86)

#set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G")

#set(CUDAToolkit_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9")
find_package(CUDAToolkit REQUIRED)
#find_package(cudss REQUIRED) # COMPONENTS cudss)

#set(CUDSS_DIR "C:/Program Files/NVIDIA cuDSS/v0.7" CACHE PATH "Path to cuDSS")

#if (MSVC)
#  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
#endif()

# Include directories
#include_directories("${PROJECT_SOURCE_DIR}/src")
#include_directories("${PROJECT_SOURCE_DIR}/external/eigen")
#include_directories("${CUDSS_DIR}/include")

add_library(Eigen3 INTERFACE)
target_include_directories(Eigen3 INTERFACE ${PROJECT_SOURCE_DIR}/external/eigen)

option(ENABLE_SPECTRA "Enable Spectra Eigenvalue solver" OFF)
if (ENABLE_SPECTRA)
  add_library(Spectra INTERFACE)
  target_include_directories(Spectra INTERFACE ${PROJECT_SOURCE_DIR}/external/spectra/include)
endif()

# =========================
# AMGX option switch
# =========================
option(ENABLE_AMGX "Enable NVIDIA AMGX support" OFF)

if (ENABLE_AMGX)
  include(ExternalProject)
  
  set(AMGX_SOURCE_DIR ${PROJECT_SOURCE_DIR}/external/amgx)
  set(AMGX_BINARY_DIR ${CMAKE_BINARY_DIR}/_deps/amgx-build)
  
  ExternalProject_Add(amgx_ep
    SOURCE_DIR  ${AMGX_SOURCE_DIR}
    BINARY_DIR  ${AMGX_BINARY_DIR}
    CMAKE_ARGS
      -DCMAKE_BUILD_TYPE=Release
      -DCMAKE_NO_MPI=TRUE
      -DCMAKE_CUDA_ARCHITECTURES="80;86"
    INSTALL_COMMAND ""
  )
  
  add_library(AMGX::amgxsh SHARED IMPORTED GLOBAL)
  add_dependencies(AMGX::amgxsh amgx_ep)
  
  if(WIN32)
    set_target_properties(AMGX::amgxsh PROPERTIES
      IMPORTED_LOCATION "${AMGX_BINARY_DIR}/Release/amgxsh.dll"
      IMPORTED_IMPLIB "${AMGX_BINARY_DIR}/Release/amgxsh.lib"
      INTERFACE_INCLUDE_DIRECTORIES "${AMGX_SOURCE_DIR}/include"
    )
  else()
    set_target_properties(AMGX::amgxsh PROPERTIES
      IMPORTED_LOCATION "${AMGX_BINARY_DIR}/libamgx.a"
      INTERFACE_INCLUDE_DIRECTORIES "${AMGX_SOURCE_DIR}/include"
    )
  endif()
endif()

option(EIGEN_USE_MKL_ALL "Use MKL for all Eigen operations" OFF)
if (EIGEN_USE_MKL_ALL)
  find_package(MKL CONFIG REQUIRED PATHS $ENV{MKLROOT})
#  target_link_libraries(Eigen3 INTERFACE MKL::MKL)
endif()

add_subdirectory(src)
add_subdirectory(example)
