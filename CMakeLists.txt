cmake_minimum_required(VERSION 3.5.1)
project(crc-hamming-distance)

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
SET(CUDA_SEPARABLE_COMPILATION ON)

find_package(CUDA 8.0)

file(GLOB_RECURSE sources
    "${CMAKE_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/*.cu"
)
set(main_source "${CMAKE_SOURCE_DIR}/src/main.cu")
list(REMOVE_ITEM sources ${main_source})
file(GLOB_RECURSE test_sources
    "${CMAKE_SOURCE_DIR}/test/*.cpp"
    "${CMAKE_SOURCE_DIR}/test/*.cu"
)

CUDA_INCLUDE_DIRECTORIES("${CMAKE_SOURCE_DIR}/include")
CUDA_ADD_EXECUTABLE(search ${main_source} ${sources})
CUDA_ADD_EXECUTABLE(tests ${test_sources} ${sources})

add_definitions(-DUSE_CUDA)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}) 
