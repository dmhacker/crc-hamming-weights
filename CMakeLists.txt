cmake_minimum_required(VERSION 3.5.1)
project(crc-hamming-distance)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

include(CTest)
include(cmake/Catch.cmake)
find_package(CUDA 8.0)

file(GLOB_RECURSE sources
    "${CMAKE_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/*.cu"
)
set(main_source "${CMAKE_SOURCE_DIR}/src/main.cpp")
list(REMOVE_ITEM sources ${main_source})
file(GLOB_RECURSE test_sources
    "${CMAKE_SOURCE_DIR}/test/*.cpp"
    "${CMAKE_SOURCE_DIR}/test/*.cu"
)

CUDA_INCLUDE_DIRECTORIES("${CMAKE_SOURCE_DIR}/include")
CUDA_ADD_LIBRARY(crcham SHARED ${sources})
CUDA_ADD_EXECUTABLE(search ${main_source})
target_link_libraries(search crcham)
CUDA_ADD_EXECUTABLE(tests ${test_sources})
target_include_directories(tests PUBLIC "${CMAKE_SOURCE_DIR}/include")
target_link_libraries(tests crcham)

add_definitions(-DUSE_CUDA)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}) 

catch_discover_tests(tests)
