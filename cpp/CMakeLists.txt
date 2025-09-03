cmake_minimum_required(VERSION 3.18)
project(mt_infer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED ONNXRUNTIME_DIR)
  message(FATAL_ERROR "Please set -DONNXRUNTIME_DIR=/path/to/onnxruntime")
endif()

set(ONNXRUNTIME_INCLUDE_DIR ${ONNXRUNTIME_DIR}/include)
set(ONNXRUNTIME_LIB_DIR ${ONNXRUNTIME_DIR}/lib)
include_directories(${ONNXRUNTIME_INCLUDE_DIR})
link_directories(${ONNXRUNTIME_LIB_DIR})

find_package(PkgConfig REQUIRED)
pkg_check_modules(SENTENCEPIECE REQUIRED sentencepiece)
include_directories(${SENTENCEPIECE_INCLUDE_DIRS})

add_executable(mt_infer main.cpp)
target_link_libraries(mt_infer onnxruntime ${SENTENCEPIECE_LIBRARIES})