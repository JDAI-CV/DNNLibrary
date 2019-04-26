# dnn_protobuf_generate_cpp(ONNX_SRCS ONNX_HDRS onnx.proto3)

set(dnnlibrary_src 
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
if (DNN_READ_ONNX)
    list(APPEND dnnlibrary_src 
        src/OnnxReader.cpp
        src/OnnxReader.h)
endif()

add_library(dnnlibrary
    ${dnnlibrary_src}
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

if (DNN_READ_ONNX)
    target_link_libraries(
        dnnlibrary
        onnx
        )
endif()


treat_warnings_as_errors(dnnlibrary)

if (DNN_BUILD_JNI)
    add_library(
        daq-jni
        SHARED
        src/JavaWrapper.cpp)
    target_link_libraries(daq-jni
        dnnlibrary)
endif()
