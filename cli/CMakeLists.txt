if (NOT IS_DIRECTORY "${CMAKE_SOURCE_DIR}/third_party/cxxopts/include")
  message(FATAL_ERROR "The client dependency repository (cxxopts) is missing! "
    "You probably didn't clone the project with --recursive. You can include it "
    "by calling \"git submodule update --init --recursive\"")
endif()

include_directories(
  ${CMAKE_SOURCE_DIR}/third_party/cxxopts/include
  )

add_executable(translate
  translate.cc
  )
target_link_libraries(translate PRIVATE
  ${PROJECT_NAME}
  )

install(
  TARGETS translate
  DESTINATION ${CMAKE_INSTALL_BINDIR}
  )
