cmake_minimum_required(VERSION 3.5.2)
project(DNNLibrary)

option(DNN_BUILD_BIN "Build binaries" ON)
option(DNN_BUILD_JNI "Build Java Wrapper" OFF)
option(DNN_SYSTEM_PROTOBUF "Use system protobuf when building onnx2daq" OFF)

include(cmake/system.cmake)
include(cmake/glog.cmake)
include(cmake/common.cmake)

configure_glog()

if (${CMAKE_SYSTEM_NAME} STREQUAL "Android")
    set (CMAKE_CXX_STANDARD 17)

    add_compile_options(-Os -fno-rtti)
    add_subdirectory(dnnlibrary)
    add_subdirectory(binaries)
else()
    set (CMAKE_CXX_STANDARD 11)
    include(cmake/onnx.cmake)
    configure_onnx()
    add_subdirectory(tools)
endif()
