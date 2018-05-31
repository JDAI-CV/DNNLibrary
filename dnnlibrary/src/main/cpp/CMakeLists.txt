# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.4.1)

set (CMAKE_CXX_STANDARD 11)

include_directories(${CMAKE_SYSROOT}/usr/include)

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.

add_library( # Sets the name of the library.
             dnnlibrary

             # Sets the library as a shared library.
            STATIC

             # Provides a relative path to your source file(s).
             ModelBuilder.cpp
             ModelBuilder.h
             Model.cpp
             Model.h nnapi_benchmark.cpp android_log_helper.h)

# Searches for a specified prebuilt library and stores the path as a
# variable. Because CMake includes system libraries in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.

find_library( # Sets the name of the path variable.
        android-lib

        # Specifies the name of the NDK library that
        # you want CMake to locate.
        android )

find_library( # Sets the name of the path variable.
              log-lib

              # Specifies the name of the NDK library that
              # you want CMake to locate.
              log )

find_library( neural-networks-lib
              neuralnetworks)

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.

target_link_libraries( # Specifies the target library.
                       dnnlibrary

                       # Links the target library to the log library
                       # included in the NDK.

                       ${log-lib}
        ${android-lib}
                       ${neural-networks-lib})

add_executable(nnapi_benchmark
        nnapi_benchmark.cpp)

target_link_libraries(nnapi_benchmark
        dnnlibrary)

add_executable(dnn_infer_simple
        stb_image.h
        dnn_infer_simple.cpp)

target_link_libraries(dnn_infer_simple
        dnnlibrary)

add_executable(dnn_simplest
        dnn_simplest.cpp)

target_link_libraries(dnn_simplest
        dnnlibrary)
