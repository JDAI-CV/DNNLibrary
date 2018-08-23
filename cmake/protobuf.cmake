macro(build_protobuf)
    message(STATUS "Building protobuf...")
    option(protobuf_BUILD_TESTS "" OFF)
    option(protobuf_BUILD_EXAMPLES "" OFF)
    option(protobuf_BUILD_SHARED_LIBS "" OFF)
    option(protobuf_BUILD_PROTOC_BINARIES "" OFF)
    add_subdirectory(${PROJECT_SOURCE_DIR}/protobuf/cmake)
endmacro()

################################################################################################
# Modified from https://github.com/pytorch/pytorch/blob/master/cmake/ProtoBuf.cmake and FindProtobuf.cmake in cmake modules
# Redefinition of protobuf_generate_cpp() for support proto3
# Usage:
#   dnn_protobuf_generate_cpp(<srcs_var> <hdrs_var> <proto_file>)
function(dnn_protobuf_generate_cpp srcs_var hdrs_var)
    if(NOT ARGN)
        message(SEND_ERROR "Error: dnn_protobuf_generate_cpp() called without any proto files")
        return()
    endif()

    set(${srcs_var})
    set(${hdrs_var})
    foreach(fil ${ARGN})
        message("Compiling ${fil}")
        get_filename_component(abs_fil ${fil} ABSOLUTE)
        get_filename_component(fil_we ${fil} NAME_WE)
        get_filename_component(ext ${fil} EXT)
        if (".proto3" STREQUAL ".proto3")
            set(src_fn "${CMAKE_CURRENT_BINARY_DIR}/${fil}.pb.cc")
            set(hdr_fn "${CMAKE_CURRENT_BINARY_DIR}/${fil}.pb.h")
        elseif(${ext} STREQUAL ".proto")
            set(src_fn "${CMAKE_CURRENT_BINARY_DIR}/${fil_we}.pb.cc")
            set(hdr_fn "${CMAKE_CURRENT_BINARY_DIR}/${fil_we}.pb.h")
        else()
            message(FATAL_ERROR
                    "Invalid proto file ${fil}")
        endif()
        list(APPEND ${srcs_var} ${src_fn})
        list(APPEND ${hdrs_var} ${hdr_fn})

        add_custom_command(
                OUTPUT ${src_fn}
                ${hdr_fn}
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}"
                COMMAND protobuf::protoc -I${PROJECT_SOURCE_DIR} --cpp_out=${DLLEXPORT_STR}${PROJECT_BINARY_DIR} ${abs_fil}
                DEPENDS protobuf::protoc ${abs_fil}
                COMMENT "Running C++ protocol buffer compiler on ${fil}" VERBATIM )
    endforeach()

    set_source_files_properties(${${srcs_var}} ${${hdrs_var}} PROPERTIES GENERATED TRUE)
    set(${srcs_var} ${${srcs_var}} PARENT_SCOPE)
    set(${hdrs_var} ${${hdrs_var}} PARENT_SCOPE)
endfunction()
