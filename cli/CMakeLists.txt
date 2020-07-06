if (NOT IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/cxxopts/include")
  message(FATAL_ERROR "The client dependency repository (cxxopts) is missing! "
    "You probably didn't clone the project with --recursive. You can include it "
    "by calling \"git submodule update --init --recursive\"")
endif()

add_executable(translate
  translate.cc
  )
target_include_directories(translate
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/cxxopts/include
  )
target_link_libraries(translate
  PRIVATE ${PROJECT_NAME}
  )

install(
  TARGETS translate
  DESTINATION ${CMAKE_INSTALL_BINDIR}
  )
