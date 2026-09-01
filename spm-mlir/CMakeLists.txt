cmake_minimum_required(VERSION 3.20.0)
project(spm-mlir LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

find_package(MLIR REQUIRED CONFIG)

set(LLVM_LIT_ARGS "-sv" CACHE STRING "Default options for lit")

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

set(SPM_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(SPM_BINARY_DIR ${PROJECT_BINARY_DIR})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${SPM_SOURCE_DIR}/include)
include_directories(${SPM_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include)
add_subdirectory(lib)
if(MLIR_INCLUDE_TESTS)
  add_subdirectory(test)
endif()
add_subdirectory(spm-opt)
