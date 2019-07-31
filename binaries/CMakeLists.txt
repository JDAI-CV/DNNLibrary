if (DNN_BUILD_BIN)
    add_executable(dnn_retrieve_result
        dnn_retrieve_result.cpp)
    target_link_libraries(dnn_retrieve_result
        dnnlibrary)
    target_include_directories(dnn_retrieve_result
        PRIVATE
        ${PROJECT_SOURCE_DIR}
        )

    if ((NOT DNN_READ_ONNX) OR DNN_SYSTEM_PROTOBUF)
        treat_warnings_as_errors(dnn_retrieve_result)
    endif()

    add_executable(dnn_benchmark
        dnn_benchmark.cpp)
    target_link_libraries(dnn_benchmark
        dnnlibrary)
    target_include_directories(dnn_benchmark
        PRIVATE
        ${PROJECT_SOURCE_DIR}
        )

    if ((NOT DNN_READ_ONNX) OR DNN_SYSTEM_PROTOBUF)
        treat_warnings_as_errors(dnn_benchmark)
    endif()

    add_executable(ex_model_builder
        ex_model_builder.cpp)
    target_link_libraries(ex_model_builder
        dnnlibrary)
    target_include_directories(ex_model_builder
        PRIVATE
        ${PROJECT_SOURCE_DIR}
        )

    if ((NOT DNN_READ_ONNX) OR DNN_SYSTEM_PROTOBUF)
        treat_warnings_as_errors(ex_model_builder)
    endif()

    add_executable(get_devices
        get_devices.cpp)
    target_link_libraries(get_devices
        dnnlibrary)
    target_include_directories(get_devices
        PRIVATE
        ${PROJECT_SOURCE_DIR}
        )

endif()
