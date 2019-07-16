add_executable(get_supported_nodes
    getsupportednodes.cpp)
target_link_libraries(get_supported_nodes
    onnx2daq)
target_include_directories(get_supported_nodes
    PRIVATE
    ${PROJECT_SOURCE_DIR}
    )
if (DNN_SYSTEM_PROTOBUF)
    treat_warnings_as_errors(get_supported_nodes)
endif()

