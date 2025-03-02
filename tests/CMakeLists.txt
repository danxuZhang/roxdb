add_executable(tests
    version.cc
    crud.cc
    scan.cc
)

target_link_libraries(tests
    PRIVATE
        ${PROJECT_NAME}
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(tests)