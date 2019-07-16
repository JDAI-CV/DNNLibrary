add_subdirectory(onnx2daq)
if (NOT DEFINED ONNX2DAQ_ONLY_LIB OR NOT ${ONNX2DAQ_ONLY_LIB})
    add_subdirectory(getsupportednodes)
endif()
