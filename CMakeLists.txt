cmake_minimum_required(VERSION 3.18)
set(PROJECT_NAME "roxdb")
project(${PROJECT_NAME} 
    VERSION 0.1
    DESCRIPTION "Rocks X DB"
    LANGUAGES CXX
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_STATIC "Build static library" ON)
option(BUILD_SHARED "Build shared library" OFF)

option(BUILD_EXAMPLES "Build examples" ON)
if (BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

find_package(HDF5 REQUIRED COMPONENTS CXX)
include_directories(${HDF5_INCLUDE_DIRS})

option(BUILD_BENCH "Build benchmarks" ON)
if (BUILD_BENCH)
    add_subdirectory(benchmarks/image_features)
endif()

option(USE_BUNDLED_ROCKSDB "Use bundled RocksDB" ON)
if(USE_BUNDLED_ROCKSDB)
    message(STATUS "Using bundled RocksDB")
    set(WITH_GFLAGS OFF CACHE BOOL "build with GFlags")
    set(WITH_TESTS OFF CACHE BOOL "build with tests")
    set(WITH_TOOLS OFF CACHE BOOL "build with tools")
    set(WITH_BENCHMARK_TOOLS OFF CACHE BOOL "build with benchmark tools")
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_party/rocksdb EXCLUDE_FROM_ALL)
    add_library(RocksDB::rocksdb ALIAS rocksdb)
else()
    find_package(RocksDB REQUIRED)
endif()

option(BUILD_TESTS "Build tests" ON)
if (BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.16.0  
    )
    FetchContent_MakeAvailable(googletest)
    enable_testing()
    add_subdirectory(tests)
endif()

option(USE_BUNDLED_FAISS "Use bundled FAISS" ON)
if(USE_BUNDLED_FAISS)
    message(STATUS "Using bundled FAISS")
    set(FAISS_ENABLE_GPU OFF CACHE BOOL "build with GPU support")
    set(FAISS_ENABLE_PYTHON OFF CACHE BOOL "build python bindings")
    set(BUILD_TESTING OFF CACHE BOOL "build tests")
    set(FAISS_OPT_LEVEL avx512 CACHE STRING "FAISS optimization level")
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_party/faiss EXCLUDE_FROM_ALL)
    add_library(FAISS::faiss ALIAS faiss)
else()
    find_package(FAISS REQUIRED)
endif()

option(USE_BUNDLED_FLATBUFFERS "Use bundled FlatBuffers" ON)
if(USE_BUNDLED_FLATBUFFERS)
    message(STATUS "Using bundled FlatBuffers")
    set(FLATBUFFERS_BUILD_TESTS OFF CACHE BOOL "Disable FlatBuffers tests")
    set(FLATBUFFERS_BUILD_FLATC ON CACHE BOOL "Enable FlatBuffers compiler")
    set(FLATBUFFERS_BUILD_FLATHASH OFF CACHE BOOL "Disable FlatBuffers hash utility")
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_party/flatbuffers EXCLUDE_FROM_ALL)
    add_library(FlatBuffers::flatbuffers ALIAS flatbuffers)
    # Make flatc compiler available
    set(FLATC_EXECUTABLE $<TARGET_FILE:flatc>)
else()
    find_package(FlatBuffers REQUIRED)
    find_program(FLATC_EXECUTABLE flatc)
    if(NOT FLATC_EXECUTABLE)
        message(FATAL_ERROR "FlatBuffers compiler (flatc) not found")
    endif()
endif()

option(USE_AVX512 "Use AVX512 SIMD" ON)
if (USE_AVX512)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__AVX512F__ -mavx512f")
endif()

option(USE_OPENMP "Use OpenMP" ON)
if (USE_OPENMP)
    find_package(OpenMP REQUIRED)  
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -DUSE_OPENMP")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

include_directories(include)
file(GLOB_RECURSE SOURCES
        src/*.h
        src/*.cc
)

if(BUILD_STATIC)
    add_library(${PROJECT_NAME} STATIC ${SOURCES})
elseif(BUILD_SHARED)
    add_library(${PROJECT_NAME} SHARED ${SOURCES})
endif()

target_include_directories(${PROJECT_NAME}
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(${PROJECT_NAME}
        PUBLIC
        RocksDB::rocksdb
        FlatBuffers::flatbuffers
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
endif()
