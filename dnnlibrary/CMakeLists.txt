# dnn_protobuf_generate_cpp(ONNX_SRCS ONNX_HDRS onnx.proto3)

add_library(dnnlibrary
    include/ModelBuilder.h
    include/Model.h
    include/DaqReader.h
    include/android_log_helper.h
    include/operand_helper.h
    include/flatbuffers_helper.h
    src/ModelBuilder.cpp
    src/Model.cpp
    src/DaqReader.cpp 
    ${PROJECT_SOURCE_DIR}/common/Shaper.h
    ${PROJECT_SOURCE_DIR}/common/Shaper.cpp
    ${PROJECT_SOURCE_DIR}/common/StrKeyMap.h 
    )

target_include_directories(
    dnnlibrary
    PUBLIC
    $<INSTALL_INTERFACE:include>
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    ${CMAKE_SYSROOT}/usr/include
    ${CMAKE_CURRENT_BINARY_DIR})

find_library(
    android-lib
    android 
    )

find_library(
    log-lib
    log
    )

find_library(
    neural-networks-lib
    neuralnetworks
    )

target_link_libraries(
    dnnlibrary
    glog::glog
    ${android-lib}
    ${log-lib}
    ${neural-networks-lib}
    )

treat_warnings_as_errors(dnnlibrary)

if (BUILD_JNI)
    add_library(
        dnn-jni
        SHARED
        src/JavaWrapper.cpp)
    target_link_libraries(dnn-jni
        dnnlibrary)
endif()
