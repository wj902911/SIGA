# Automatically find all source and header files in the src directory
file(GLOB_RECURSE SRC_FILES_CPP "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
file(GLOB_RECURSE SRC_FILES_CU "${CMAKE_CURRENT_SOURCE_DIR}/*.cu" "${CMAKE_CURRENT_SOURCE_DIR}/*.cuh")

# Create an object library for all source files
add_library(SIGA STATIC ${SRC_FILES_CPP} ${SRC_FILES_CU})
set_property(TARGET SIGA PROPERTY FOLDER "library")

# Include directories for the source files
target_include_directories(SIGA PUBLIC "${PROJECT_SOURCE_DIR}/src")