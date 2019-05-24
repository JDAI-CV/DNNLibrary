function(configure_onnx)
    if (TARGET onnx)
        return()
    endif()

    if (NOT ${DNN_SYSTEM_PROTOBUF})
        include(${PROJECT_SOURCE_DIR}/cmake/protobuf.cmake)
        configure_protobuf()
    endif()

    message(STATUS "Configuring onnx...")
    set(DAQ_ONNX_NAMESPACE onnx_daq)
    if (MSVC)
        set(ONNX_CMAKELISTS ${PROJECT_SOURCE_DIR}/third_party/onnx/CMakeLists.txt)
        file(READ ${ONNX_CMAKELISTS} content)
        string(
            REPLACE
            "/WX"
            ""
            content
            "${content}"
            )
        file(WRITE ${ONNX_CMAKELISTS} "${content}")
    endif()
    set(ONNX_BUILD_MAIN_LIB ON)
    set(ONNX_NAMESPACE ${DAQ_ONNX_NAMESPACE} CACHE STRING "onnx namespace")
    if (${CMAKE_SYSTEM_NAME} STREQUAL "Android" AND NOT EXISTS ${DNN_CUSTOM_PROTOC_EXECUTABLE})
        message(FATAL ERROR "DNN_CUSTOM_PROTOC_EXECUTABLE is not set or wrong.")
    endif()
    set(ONNX_CUSTOM_PROTOC_EXECUTABLE ${DNN_CUSTOM_PROTOC_EXECUTABLE})
    option(ONNX_USE_LITE_PROTO "" ON)
    add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/onnx)
    target_compile_definitions(onnx_proto PRIVATE ONNX_BUILD_MAIN_LIB)
    # Since https://github.com/onnx/onnx/pull/1318 is merged, we don't need to set it manually
    # target_compile_definitions(onnx
    # PUBLIC
    # -DONNX_NAMESPACE=${DAQ_ONNX_NAMESPACE})
endfunction()
