# Automatically find all .cpp files in the example directory
file(GLOB_RECURSE EXAMPLE_FILES_CPP "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
file(GLOB_RECURSE EXAMPLE_FILES_CU "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")

# Combine all example files
set(EXAMPLE_FILES ${EXAMPLE_FILES_CPP} ${EXAMPLE_FILES_CU})

# Add an executable for each .cpp file found
foreach(EXAMPLE_FILE ${EXAMPLE_FILES})
    # Get the filename without the extension and directory
    get_filename_component(EXAMPLE_NAME ${EXAMPLE_FILE} NAME_WE)

    # Add the executable
    add_executable(${EXAMPLE_NAME} ${EXAMPLE_FILE})
    set_property(TARGET ${EXAMPLE_NAME} PROPERTY FOLDER "example")

    target_link_libraries(${EXAMPLE_NAME} SIGA ${GLFW_LIBRARIES})

endforeach()