# dnn_protobuf_generate_cpp(ONNX_PROTO_SRCS ONNX_PROTO_HDRS onnx.proto3)
# message(${ONNX_PROTO_SRCS})
# message(${CMAKE_CURRENT_BINARY_DIR})
add_library(onnx2daq
    ${CMAKE_CURRENT_SOURCE_DIR}/OnnxConverter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/OnnxConverterImpl.cpp
    ${PROJECT_SOURCE_DIR}/include/tools/onnx2daq/OnnxConverter.h
    ${CMAKE_CURRENT_SOURCE_DIR}/NodeAttrHelper.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/NodeAttrHelper.h
    ${PROJECT_SOURCE_DIR}/include/common/StrKeyMap.h
    ${PROJECT_SOURCE_DIR}/include/common/Shaper.h
    ${PROJECT_SOURCE_DIR}/common/Shaper.cpp
    ${PROJECT_SOURCE_DIR}/common/internal_vars.cpp
    ${PROJECT_SOURCE_DIR}/common/internal_vars.h
    )
target_link_libraries(onnx2daq
    flatbuffers
    protobuf::libprotobuf-lite
    glog::glog
    onnx)
target_include_directories(onnx2daq
    PUBLIC
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    )
if (${DNN_SYSTEM_PROTOBUF})
    # protobuf headers in third_party brings many warnings, need 
    # a way to suppress these warnings
    # before getting a proper way, only enable -Werror when using system protobuf
    treat_warnings_as_errors(onnx2daq)
endif()

if (DNN_CMAKE_INSTALL)
    include(GNUInstallDirs)
    if (CMAKE_SYSTEM_NAME STREQUAL "Android")
        # Build DNNLibrary (on Android)
        install(TARGETS onnx2daq
            EXPORT DNNLibraryTargets
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            )
    else()
        # Build onnx2daq (on non-Android)
        install(TARGETS onnx2daq
            EXPORT ONNX2daqTargets
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            )
    endif()
endif()
add_library(dnnlibrary::onnx2daq ALIAS onnx2daq)

if (NOT ONNX2DAQ_ONLY_LIB)
    # Build onnx2daq-bin (on non-Android)
    add_executable(onnx2daq-bin
        ${CMAKE_CURRENT_SOURCE_DIR}/onnx2daq.cpp)
    target_link_libraries(onnx2daq-bin
        onnx2daq)
    if (DNN_SYSTEM_PROTOBUF)
        treat_warnings_as_errors(onnx2daq-bin)
    endif()
    set_target_properties(onnx2daq-bin PROPERTIES OUTPUT_NAME "onnx2daq")
    if (DNN_CMAKE_INSTALL)
        install(TARGETS onnx2daq-bin
            EXPORT ONNX2daqTargets
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    endif()
    
	add_executable(dnnlibrary::onnx2daq-bin ALIAS onnx2daq-bin)

    if (${DNN_BUILD_PYTHON})
        pybind11_add_module(_onnx2daq ${CMAKE_CURRENT_SOURCE_DIR}/pywrapper.cpp)

        target_link_libraries(_onnx2daq
            PUBLIC
            onnx2daq
            pybind11::pybind11
            )
    endif()
endif()

if (NOT ${CMAKE_SYSTEM_NAME} STREQUAL "Android")
    if (DNN_CMAKE_INSTALL)
        install(EXPORT ONNX2daqTargets 
            NAMESPACE dnnlibrary:: 
            DESTINATION share/cmake/ONNX2daq
            )

        configure_file("${PROJECT_SOURCE_DIR}/cmake/ONNX2daqConfig.cmake.in"
            "${PROJECT_BINARY_DIR}/cmake/ONNX2daqConfig.cmake"
            @ONLY)

        install(FILES "${PROJECT_BINARY_DIR}/cmake/ONNX2daqConfig.cmake" DESTINATION share/cmake/ONNX2daq)
    endif()
endif()
