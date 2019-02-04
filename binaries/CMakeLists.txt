if (DNN_BUILD_BIN)
    add_executable(dnn_retrieve_result
        dnn_retrieve_result.cpp)
    target_link_libraries(dnn_retrieve_result
        dnnlibrary)

    treat_warnings_as_errors(dnn_retrieve_result)

    add_executable(dnn_benchmark
        dnn_benchmark.cpp)
    target_link_libraries(dnn_benchmark
        dnnlibrary)

    treat_warnings_as_errors(dnn_benchmark)

    add_executable(ex_model_builder
        ex_model_builder.cpp)
    target_link_libraries(ex_model_builder
        dnnlibrary)

    treat_warnings_as_errors(ex_model_builder)
endif()
