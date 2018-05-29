enable_testing()
find_package(GTest)

if (GTEST_FOUND)
  find_package(Threads)

  add_executable(onmt_test
    ../compute_mkl.cc
    ../compute_simd.cc
    test.cc)
  target_include_directories(onmt_test PUBLIC
    ${GTEST_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}
    )
  target_link_libraries(onmt_test
    ${GTEST_BOTH_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
    ${MKL_LIBRARIES} # TODO: link to own library
    )
  #add_test(TokenizerTest onmt_tokenizer_test "${CMAKE_CURRENT_LIST_DIR}/data/")
endif()
