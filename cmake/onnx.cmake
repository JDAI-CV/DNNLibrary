macro(configure_onnx)
    message(STATUS "Configuring onnx...")
    set(DAQ_ONNX_NAMESPACE onnx_daq)
    
    set(ONNX_NAMESPACE ${DAQ_ONNX_NAMESPACE} CACHE STRING "onnx namespace")
    add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/onnx)
    # Since https://github.com/onnx/onnx/pull/1318 is merged, we don't need to set it manually
    # target_compile_definitions(onnx
    # PUBLIC
    # -DONNX_NAMESPACE=${DAQ_ONNX_NAMESPACE})
endmacro()
