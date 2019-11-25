# dnn_protobuf_generate_cpp(ONNX_SRCS ONNX_HDRS onnx.proto3)

set(dnnlibrary_src 
    ${PROJECT_SOURCE_DIR}/include/dnnlibrary/ModelBuilder.h
    ${PROJECT_SOURCE_DIR}/include/dnnlibrary/Model.h
    ${PROJECT_SOURCE_DIR}/include/dnnlibrary/DaqReader.h
    ${PROJECT_SOURCE_DIR}/include/dnnlibrary/nnapi_implementation.h
    android_log_helper.h
    flatbuffers_helper.h
    ModelBuilder.cpp
    ModelBuilderImpl.cpp
    Model.cpp
    DaqReader.cpp 
    NeuralNetworksWrapper.cpp
    nnapi_implementation.cc
    ${PROJECT_SOURCE_DIR}/include/common/Shaper.h
    ${PROJECT_SOURCE_DIR}/common/Shaper.cpp
    ${PROJECT_SOURCE_DIR}/include/common/StrKeyMap.h 
    ${PROJECT_SOURCE_DIR}/common/internal_vars.cpp
    ${PROJECT_SOURCE_DIR}/common/internal_vars.h
    )
if (DNN_READ_ONNX)
    list(APPEND dnnlibrary_src 
        OnnxReader.cpp
        ${PROJECT_SOURCE_DIR}/include/dnnlibrary/OnnxReader.h)
endif()

add_library(dnnlibrary
    ${dnnlibrary_src}
    )

target_include_directories(
    dnnlibrary
    PUBLIC
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR})

find_library(
    android-lib
    android 
    )

find_library(
    log-lib
    log
    )

find_library(
    neural-networks-lib
    neuralnetworks
    )

target_link_libraries(
    dnnlibrary
    glog::glog
    flatbuffers
    ${android-lib}
    ${log-lib}
    ${neural-networks-lib}
    )

if (DNN_READ_ONNX)
    target_link_libraries(
        dnnlibrary
        onnx2daq
        )
    target_compile_definitions(
        dnnlibrary
        PUBLIC
        DNN_READ_ONNX
        )
else()
    # Reading ONNX need protobuf, which produces warnings
    treat_warnings_as_errors(dnnlibrary)
endif()

if (DNN_CMAKE_INSTALL)
    include(GNUInstallDirs)
    install(TARGETS dnnlibrary
        EXPORT DNNLibraryTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        )
    install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/
            DESTINATION include)
endif()
add_library(dnnlibrary::dnnlibrary ALIAS dnnlibrary)

if (DNN_BUILD_JNI)
    add_library(
        daq-jni
        SHARED
        JavaWrapper.cpp)
    target_link_libraries(daq-jni
        dnnlibrary)
    if (DNN_CMAKE_INSTALL)
        install(TARGETS daq-jni
            EXPORT DNNLibraryTargets
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
    endif()
    add_library(dnnlibrary::daq-jni ALIAS daq-jni)
endif()

if (DNN_CMAKE_INSTALL)
    install(EXPORT DNNLibraryTargets 
        NAMESPACE dnnlibrary:: 
        DESTINATION share/cmake/DNNLibrary
        )

    configure_file("${PROJECT_SOURCE_DIR}/cmake/DNNLibraryConfig.cmake.in"
        "${PROJECT_BINARY_DIR}/cmake/DNNLibraryConfig.cmake"
        @ONLY)

    install(FILES "${PROJECT_BINARY_DIR}/cmake/DNNLibraryConfig.cmake" DESTINATION share/cmake/DNNLibrary)
endif()
