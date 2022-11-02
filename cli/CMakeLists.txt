if (NOT IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/cxxopts/include")
  message(FATAL_ERROR "The client dependency repository (cxxopts) is missing! "
    "You probably didn't clone the project with --recursive. You can include it "
    "by calling \"git submodule update --init --recursive\"")
endif()

add_executable(translator
  translator.cc
  )
target_include_directories(translator
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/cxxopts/include
  )
target_link_libraries(translator
  PRIVATE ${PROJECT_NAME}
)

set_target_properties(translator PROPERTIES OUTPUT_NAME ct2-translator)

install(
  TARGETS translator
  DESTINATION ${CMAKE_INSTALL_BINDIR}
  )
