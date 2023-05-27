################################################################################

# WARNING: to list the system libraries(ie IMPORTED) you MUST set:
# set_target_properties(your_lib PROPERTIES IMPORTED_GLOBAL TRUE)
# just after the find_package call
# cf https://gitlab.kitware.com/cmake/cmake/-/issues/17256
#
# https://stackoverflow.com/questions/32756195/recursive-list-of-link-libraries-in-cmake
# https://stackoverflow.com/questions/32197663/how-can-i-remove-the-the-location-property-may-not-be-read-from-target-error-i
function(_get_link_libraries OUTPUT_LIST TARGET)
    list(APPEND VISITED_TARGETS ${TARGET})

    # DO NOT switch on IMPORTED or not
    # An INTERFACE library CAN have LINK_LIBRARIES!
    # get_target_property(IMPORTED ${TARGET} IMPORTED)
    set(LIBS "")
    get_target_property(LIBS_1 ${TARGET} INTERFACE_LINK_LIBRARIES)
    get_target_property(LIBS_2 ${TARGET} LINK_LIBRARIES)
    list(APPEND LIBS ${LIBS_1} ${LIBS_2})

    set(LIB_FILES "")

    foreach(LIB ${LIBS})
        if (TARGET ${LIB})
            list(FIND VISITED_TARGETS ${LIB} VISITED)
            if (${VISITED} EQUAL -1)
                # OLD: get_target_property(LIB_FILE ${LIB} LOCATION)
                # NEW:
                _get_link_libraries(LINK_LIB_FILES ${LIB})
                set(LIB_FILE ${LIB})
                list(APPEND LIB_FILES ${LINK_LIB_FILES})
                list(APPEND LIB_FILES ${LIB_FILE})
            endif()
        endif()
    endforeach()

    set(VISITED_TARGETS ${VISITED_TARGETS} PARENT_SCOPE)
    set(${OUTPUT_LIST} ${LIB_FILES} PARENT_SCOPE)
endfunction()

################################################################################

function(export_all_target_libs TARGET)
    # NOTE: get_target_property(CIRCUIT_LIB_LINK_LIBRARIES a_target LINK_LIBRARIES) is NOT transitive
    # This function will return eg: "$<TARGET_FILE:rust_cxx>;$<TARGET_FILE:circuit_lib>;"
    # b/c generator expression are evaluated LATER
    # cf https://stackoverflow.com/questions/59226127/cmake-generator-expression-how-to-get-target-file-property-on-list-of-targets
    set(ALL_LINK_LIBRARIES "")
    _get_link_libraries(ALL_LINK_LIBRARIES ${TARGET})

    message(STATUS "ALL_LINK_LIBRARIES : ${ALL_LINK_LIBRARIES}")

    set(ALL_LIBS "")
    # TODO move that back into get_link_libraries
    # NOTE: we MUST do it in 2 steps:
    # - collect all the LINK_LIBRARIES recursively
    # - loop on those and get their TARGET_FILE (if not INTERFACE_LIBRARY)
    # That is b/c in get_link_libraries a INTERFACE_LIBRARY CAN have link_libraries
    # but we CAN NOT evaluate generator expressions at this time.
    foreach(LIB ${ALL_LINK_LIBRARIES})
        # MUST skip INTERFACE else:
        # CMake Error at src/CMakeLists.txt:136 (add_custom_command):
        # Error evaluating generator expression:
        #   $<TARGET_FILE:rust_cxx>
        # Target "rust_cxx" is not an executable or library.
        # SHARED_LIBRARY,INTERFACE_LIBRARY,STATIC_LIBRARY
        #
        get_target_property(LIB_TYPE ${LIB} TYPE)
        message(STATUS "LIB_TYPE : ${LIB} = ${LIB_TYPE}")

        if(NOT ${LIB_TYPE} STREQUAL "INTERFACE_LIBRARY")
            set(LIB_FILE $<TARGET_FILE:${LIB}>)
            list(APPEND ALL_LIBS ${LIB_FILE})
        endif()
    endforeach()  # LIB ${ALL_LIBS}

    message(STATUS "ALL_LIBS : ${ALL_LIBS}")

    # add_custom_command(ie echoing only to stdout) works but more difficult to get from build.rs
    # b/c when there is "ninja: no work to do" it will NOT echo on the console
    add_custom_command(
        TARGET ${TARGET}
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo ${ALL_LIBS} > ${CMAKE_CURRENT_BINARY_DIR}/cmake_generated_libs
        # OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/cmake_generated_libs
        VERBATIM
    )
endfunction(export_all_target_libs)
