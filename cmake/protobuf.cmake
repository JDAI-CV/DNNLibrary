macro(build_protobuf)
    message(STATUS "Building protobuf...")
    option(protobuf_BUILD_TESTS "" OFF)
    option(protobuf_BUILD_EXAMPLES "" OFF)
    option(protobuf_BUILD_SHARED_LIBS "" OFF)
    option(protobuf_BUILD_PROTOC_BINARIES "" OFF)
    add_subdirectory(third_party/protobuf/cmake)
endmacro()

################################################################################################
# Modified from https://github.com/pytorch/pytorch/blob/master/cmake/ProtoBuf.cmake
# Redefinition of protobuf_generate_cpp() for support cross-compilation
# Usage:
#   dnn_protobuf_generate_cpp(<srcs_var> <hdrs_var> <python_var>)
function(dnn_protobuf_generate_cpp srcs_var hdrs_var)
    if(NOT ARGN)
        message(SEND_ERROR "Error: dnn_protobuf_generate_cpp() called without any proto files")
        return()
    endif()

    set(${srcs_var})
    set(${hdrs_var})
    foreach(fil ${ARGN})
        get_filename_component(abs_fil ${fil} ABSOLUTE)
        get_filename_component(fil_we ${fil} NAME_WE)
        get_filename_component(ext ${fil} EXT)
        if (".proto3" STREQUAL ".proto3")
            list(APPEND ${srcs_var} "${CMAKE_CURRENT_BINARY_DIR}/${fil}.pb.cc")
            list(APPEND ${hdrs_var} "${CMAKE_CURRENT_BINARY_DIR}/${fil}.pb.h")
        elseif(${ext} STREQUAL ".proto")
            list(APPEND ${srcs_var} "${CMAKE_CURRENT_BINARY_DIR}/${fil_we}.pb.cc")
            list(APPEND ${hdrs_var} "${CMAKE_CURRENT_BINARY_DIR}/${fil_we}.pb.h")
        else()
            message(FATAL_ERROR
                    "Invalid proto file ${fil}")
        endif()

        add_custom_command(
                OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${fil_we}.pb.cc"
                "${CMAKE_CURRENT_BINARY_DIR}/${fil_we}.pb.h"
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}"
                COMMAND ${PROTOC_EXECUTABLE} -I${PROJECT_SOURCE_DIR} --cpp_out=${DLLEXPORT_STR}${PROJECT_BINARY_DIR} ${abs_fil}
                DEPENDS ${PROTOC_EXECUTABLE} ${abs_fil}
                COMMENT "Running C++ protocol buffer compiler on ${fil}" VERBATIM )
    endforeach()

    set_source_files_properties(${${srcs_var}} ${${hdrs_var}} PROPERTIES GENERATED TRUE)
    set(${srcs_var} ${${srcs_var}} PARENT_SCOPE)
    set(${hdrs_var} ${${hdrs_var}} PARENT_SCOPE)
endfunction()