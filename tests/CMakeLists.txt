enable_testing()
find_package(GTest)

if (GTEST_FOUND)
  find_package(Threads)

  add_executable(ctranslate2_test
    storage_view_test.cc
    ops_test.cc
    transformer_test.cc
    translator_test.cc
    test.cc)
  target_include_directories(ctranslate2_test PUBLIC
    ${GTEST_INCLUDE_DIRS}
    )
  target_link_libraries(ctranslate2_test
    ${GTEST_BOTH_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
    ${PROJECT_NAME}
    )
endif()

add_executable(benchmark_ops
  benchmark_ops.cc
  )
target_link_libraries(benchmark_ops
  ${PROJECT_NAME}
  )
