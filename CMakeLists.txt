# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.5.2)

set (CMAKE_CXX_STANDARD 17)

option(BUILD_SHARED_LIBS "Build shared dnnlibrary" OFF)
option(BUILD_JNI "Build JavaWrapper" OFF)
option(BUILD_BIN "Build binaries" ON)

macro(build_protobuf)
    message(STATUS "Building protobuf...")
    option(protobuf_BUILD_TESTS "" OFF)
    option(protobuf_BUILD_EXAMPLES "" OFF)
    option(protobuf_BUILD_SHARED_LIBS "" OFF)
    add_subdirectory(third_party/protobuf/cmake)
endmacro()

build_protobuf()
add_subdirectory(src)
