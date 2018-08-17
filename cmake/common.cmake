function(treat_warnings_as_errors target)
    if(MSVC)
        target_compile_options(${target} PRIVATE "/W4")
    elseif(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
        # Update if necessary
        target_compile_options(${target} PRIVATE "-Wall")
    endif()
endfunction()
