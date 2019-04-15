if (${DNN_SYSTEM_PROTOBUF})
    find_package(Protobuf)
endif()

# dnn_protobuf_generate_cpp(ONNX_PROTO_SRCS ONNX_PROTO_HDRS onnx.proto3)
# message(${ONNX_PROTO_SRCS})
# message(${CMAKE_CURRENT_BINARY_DIR})
add_library(onnx2daq
    ${CMAKE_CURRENT_SOURCE_DIR}/OnnxConverter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/OnnxConverter.h
    ${CMAKE_CURRENT_SOURCE_DIR}/NodeAttrHelper.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/NodeAttrHelper.h
    ${PROJECT_SOURCE_DIR}/common/StrKeyMap.h
    ${PROJECT_SOURCE_DIR}/common/Shaper.h
    ${PROJECT_SOURCE_DIR}/common/Shaper.cpp
    ${ONNX_PROTO_SRCS}
    ${ONNX_PROTO_HDRS})
target_link_libraries(onnx2daq
    protobuf::libprotobuf
    glog::glog
    onnx)
target_include_directories(onnx2daq
    PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    )
if (${DNN_SYSTEM_PROTOBUF})
    # protobuf headers in third_party brings many warnings, need 
    # a way to suppress these warnings
    # before getting a proper way, only enable -Werror when using system protobuf
    treat_warnings_as_errors(onnx2daq)
endif()


if (NOT ${ONNX2DAQ_ONLY_LIB})
    add_executable(onnx2daq-bin
        ${CMAKE_CURRENT_SOURCE_DIR}/onnx2daq.cpp)
    target_link_libraries(onnx2daq-bin
        onnx2daq)
    treat_warnings_as_errors(onnx2daq-bin)
    set_target_properties(onnx2daq-bin PROPERTIES OUTPUT_NAME "onnx2daq")
endif()

if (${DNN_BUILD_PYTHON})
    pybind11_add_module(_onnx2daq ${CMAKE_CURRENT_SOURCE_DIR}/pywrapper.cpp)

    target_link_libraries(_onnx2daq
        PUBLIC
        protobuf::libprotobuf
        glog::glog
        onnx
        onnx2daq
        pybind11::pybind11
        )
endif()
