cmake_minimum_required(VERSION 3.5.1)
project(crc-hamming-distance)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

find_package(CUDA 8.0)

file(GLOB_RECURSE sources
    "${CMAKE_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/*.cu"
)

CUDA_INCLUDE_DIRECTORIES("${CMAKE_SOURCE_DIR}/include")
CUDA_ADD_EXECUTABLE(crcham ${sources})

target_include_directories(crcham PUBLIC "${CMAKE_SOURCE_DIR}/include")
add_definitions(-DUSE_CUDA)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}) 
