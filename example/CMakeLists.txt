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

    target_link_libraries(${EXAMPLE_NAME} PRIVATE SIGA)

    if (WIN32 AND (EIGEN_USE_MKL_ALL OR ENABLE_PARDISO))
      file(GLOB SIGA_MKL_RUNTIME_DLLS
        "${MKL_ROOT}/bin/mkl_*.2.dll"
      )
      if (OMP_DLL_DIR)
        list(APPEND SIGA_MKL_RUNTIME_DLLS "${OMP_DLL_DIR}/libiomp5md.dll")
      endif()

      add_custom_command(TARGET ${EXAMPLE_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
          $<TARGET_RUNTIME_DLLS:${EXAMPLE_NAME}>
          $<TARGET_FILE_DIR:${EXAMPLE_NAME}>
        COMMAND_EXPAND_LISTS
      )

      add_custom_command(TARGET ${EXAMPLE_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${SIGA_MKL_RUNTIME_DLLS}
          $<TARGET_FILE_DIR:${EXAMPLE_NAME}>
        COMMAND_EXPAND_LISTS
      )
    endif()

    if (ENABLE_AMGX)
      target_link_libraries(${EXAMPLE_NAME} PRIVATE AMGX::amgxsh)
    
      # Only needed on Windows for the runtime DLL
      if (WIN32)
        add_custom_command(TARGET ${EXAMPLE_NAME} POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:AMGX::amgxsh>"
            "$<TARGET_FILE_DIR:${EXAMPLE_NAME}>"
        )
      endif()

      add_custom_command(TARGET ${EXAMPLE_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${PROJECT_SOURCE_DIR}/AMGX_config/SOLVER_CONFIG_INUSE.json"
          "$<TARGET_FILE_DIR:${EXAMPLE_NAME}>/SOLVER_CONFIG_INUSE.json"
      )
    endif()

endforeach()
