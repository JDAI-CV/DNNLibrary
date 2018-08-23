# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.5.2)
project(DNNLibrary)

set (CMAKE_CXX_STANDARD 17)

option(BUILD_SHARED_LIBS "Build shared dnnlibrary" OFF)
option(BUILD_JNI "Build JavaWrapper" OFF)
option(BUILD_BIN "Build binaries" ON)

include(cmake/system.cmake)
include(cmake/protobuf.cmake)
include(cmake/glog.cmake)
include(cmake/onnx.cmake)
include(cmake/common.cmake)

# build_protobuf()

# if(NOT EXISTS "${PROTOC_EXECUTABLE}")
    # if (EXISTS "${PROJECT_SOURCE_DIR}/build_host_protoc/bin/protoc")
        # set(PROTOC_EXECUTABLE ${PROJECT_SOURCE_DIR}/build_host_protoc/bin/protoc)
    # else()
        # message(FATAL_ERROR
                # "No available protoc, please run build_host_protoc.sh first")
    # endif()
# endif()

build_glog()

if (${CMAKE_SYSTEM_NAME} STREQUAL "Android")
    add_subdirectory(src)
else()
    build_onnx()
    add_subdirectory(tools)
endif()
