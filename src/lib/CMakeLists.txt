# dnn_protobuf_generate_cpp(ONNX_SRCS ONNX_HDRS onnx.proto3)

set(LIB_SRC
    include/ModelBuilder.h
    include/Model.h
    include/android_log_helper.h
    src/ModelBuilder.cpp
    src/Model.cpp
    include/operand_helper.h
    src/DaqReader.cpp src/DaqReader.h
    include/daq_generated.h ${PROJECT_SOURCE_DIR}/common/StrKeyMap.h include/flatbuffers_helper.h
    ${PROJECT_SOURCE_DIR}/common/Shaper.h
    ${PROJECT_SOURCE_DIR}/common/Shaper.cpp
    )

if (BUILD_JNI)
    list(APPEND
        LIB_SRC
        src/JavaWrapper.cpp)
endif()

add_library(
    dnnlibrary
    ${LIB_SRC})

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
    # protobuf::libprotobuf-lite
    glog::glog
    ${android-lib}
    ${log-lib}
    ${neural-networks-lib}
    )

treat_warnings_as_errors(dnnlibrary)
