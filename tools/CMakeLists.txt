add_subdirectory(onnx2daq)
if (NOT ${ONNX2DAQ_ONLY_LIB})
    add_subdirectory(getsupportednodes)
endif()
