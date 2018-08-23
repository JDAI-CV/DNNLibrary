macro(build_onnx)
    message(STATUS "Building onnx...")
    set(DAQ_ONNX_NAMESPACE onnx_daq)
    set(ONNX_NAMESPACE ${DAQ_ONNX_NAMESPACE})
    add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/onnx)
    target_compile_definitions(onnx
            PUBLIC
            -DONNX_NAMESPACE=${DAQ_ONNX_NAMESPACE})
endmacro()
