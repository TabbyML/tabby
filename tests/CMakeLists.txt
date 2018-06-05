enable_testing()
find_package(GTest)

if (GTEST_FOUND)
  find_package(Threads)

  add_executable(opennmt_test
    ops_test.cc
    test.cc)
  target_include_directories(opennmt_test PUBLIC
    ${GTEST_INCLUDE_DIRS}
    )
  target_link_libraries(opennmt_test
    ${GTEST_BOTH_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
    ${PROJECT_NAME}
    )
endif()
