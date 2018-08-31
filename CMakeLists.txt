# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.5.2)
project(DNNLibrary)

set (CMAKE_CXX_STANDARD 17)

option(BUILD_SHARED_LIBS "Build shared dnnlibrary" OFF)
option(BUILD_BIN "Build binaries" ON)
option(BUILD_JNI "Build Java Wrapper" OFF)

include(cmake/system.cmake)
include(cmake/glog.cmake)
include(cmake/common.cmake)

configure_glog()

if (${CMAKE_SYSTEM_NAME} STREQUAL "Android")
    add_compile_options(-Os -fno-rtti)
    add_subdirectory(dnnlibrary)
    add_subdirectory(binaries)
else()
    include(cmake/onnx.cmake)
    configure_onnx()
    add_subdirectory(tools)
endif()
