file(GLOB EXAMPLE_SOURCES *.cc)

# Iterate over each source file and create an executable
foreach(SOURCE_FILE ${EXAMPLE_SOURCES})
    get_filename_component(EXECUTABLE_NAME ${SOURCE_FILE} NAME_WE)
    add_executable(${EXECUTABLE_NAME} ${SOURCE_FILE})
    target_link_libraries(${EXECUTABLE_NAME} PRIVATE ${PROJECT_NAME}) 
endforeach()