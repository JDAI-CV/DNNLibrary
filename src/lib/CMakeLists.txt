set(LIB_SRC
    include/ModelBuilder.h
    include/Model.h
    include/android_log_helper.h
    src/ModelBuilder.cpp
    src/Model.cpp include/operand_helper.h)

if (BUILD_JNI)
    list(APPEND
        LIB_SRC
        src/JavaWrapper.cpp)
endif()

add_library(
    dnnlibrary
    ${LIB_SRC})

target_include_directories(
    dnnlibrary
    PUBLIC
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
    ${CMAKE_SYSROOT}/usr/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src)

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
    protobuf::libprotobuf-lite
    ${android-lib}
    ${log-lib}
    ${neural-networks-lib}
    )
