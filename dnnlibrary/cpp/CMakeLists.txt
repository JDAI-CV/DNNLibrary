# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.5.2)

set (CMAKE_CXX_STANDARD 11)

option(BUILD_SHARED_LIBS "Build shared dnnlibrary" OFF)
option(BUILD_JNI "Build JavaWrapper" OFF)
option(BUILD_BIN "Build binaries" ON)

add_subdirectory(lib)
add_subdirectory(app)
