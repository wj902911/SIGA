# Automatically find all source and header files in the src directory
file(GLOB_RECURSE SRC_FILES_CPP "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
file(GLOB_RECURSE SRC_FILES_CU "${CMAKE_CURRENT_SOURCE_DIR}/*.cu" "${CMAKE_CURRENT_SOURCE_DIR}/*.cuh")

# Create an object library for all source files
add_library(SIGA STATIC ${SRC_FILES_CPP} ${SRC_FILES_CU})
set_property(TARGET SIGA PROPERTY FOLDER "library")

# Include directories for the source files
target_include_directories(SIGA PUBLIC 
"${PROJECT_SOURCE_DIR}/src" 
#"${PROJECT_SOURCE_DIR}/external/eigen" 
#"${CUDSS_DIR}/include"
)

# Macro for your #if ENABLE_AMGX in code:
target_compile_definitions(SIGA PUBLIC
  ENABLE_AMGX=$<BOOL:${ENABLE_AMGX}>
  ENABLE_SPECTRA=$<BOOL:${ENABLE_SPECTRA}>
)

#target_link_directories(SIGA PRIVATE "${CUDSS_DIR}/lib")

target_link_libraries(SIGA
  PUBLIC         # propagate to dependents (your examples)
    Eigen3 
    CUDA::cudart
    CUDA::cusparse
    CUDA::cusolver
    CUDA::cublas
    #CUDA::cudss
    #cudss
    #cublas
    #CUDA::cudadevrt
)

if (ENABLE_SPECTRA)
  target_link_libraries(SIGA PUBLIC Spectra)
endif()

if (ENABLE_AMGX)
  target_link_libraries(SIGA PUBLIC AMGX::amgxsh)
endif()

# Optional but nice for large CUDA .cu files
#set_target_properties(SIGA PROPERTIES CUDA_SEPARABLE_COMPILATION ON)