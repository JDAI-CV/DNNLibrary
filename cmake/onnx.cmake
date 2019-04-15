macro(configure_onnx)
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
    add_compile_definitions(ONNX_BUILD_MAIN_LIB)
    set(ONNX_NAMESPACE ${DAQ_ONNX_NAMESPACE} CACHE STRING "onnx namespace")
    add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/onnx)
    # Since https://github.com/onnx/onnx/pull/1318 is merged, we don't need to set it manually
    # target_compile_definitions(onnx
    # PUBLIC
    # -DONNX_NAMESPACE=${DAQ_ONNX_NAMESPACE})
endmacro()
