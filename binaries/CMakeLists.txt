if (DNN_BUILD_BIN)
    add_executable(dnn_infer
        dnn_infer.cpp)
    target_link_libraries(dnn_infer
        dnnlibrary)

    treat_warnings_as_errors(dnn_infer)
endif()
