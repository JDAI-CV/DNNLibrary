function(configure_glog)
    if (TARGET glog::glog)
        return()
    endif()
    message(STATUS "Configureing glog...")

    set(TEMP_BUILD_TESTING ${BUILD_TESTING})
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(TEMP_WITH_GFLAGS ${WITH_GFLAGS})
    set(WITH_GFLAGS OFF CACHE BOOL "" FORCE)

    add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/glog)

    set(BUILD_TESTING ${TEMP_BUILD_TESTING} CACHE BOOL "" FORCE)
    set(WITH_GFLAGS ${TEMP_WITH_GFLAGS} CACHE BOOL "" FORCE)
endfunction()
