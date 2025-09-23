# Automatically find all source and header files in the src directory
file(GLOB_RECURSE SRC_FILES_CPP "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
file(GLOB_RECURSE SRC_FILES_CU "${CMAKE_CURRENT_SOURCE_DIR}/*.cu" "${CMAKE_CURRENT_SOURCE_DIR}/*.cuh")

# Create an object library for all source files
add_library(SIGA STATIC ${SRC_FILES_CPP} ${SRC_FILES_CU})
set_property(TARGET SIGA PROPERTY FOLDER "library")

# Include directories for the source files
target_include_directories(SIGA PUBLIC "${PROJECT_SOURCE_DIR}/src" "${PROJECT_SOURCE_DIR}/external/eigen")

target_link_libraries(SIGA
  PUBLIC         # propagate to dependents (your examples)
    CUDA::cudart
    CUDA::cusparse
    CUDA::cusolver
    #CUDA::cudadevrt
)

# Optional but nice for large CUDA .cu files
#set_target_properties(SIGA PROPERTIES CUDA_SEPARABLE_COMPILATION ON)