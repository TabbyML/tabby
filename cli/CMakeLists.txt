add_executable(translate
  translate.cc
  )
target_link_libraries(translate
  ${PROJECT_NAME}
  )

install(
  TARGETS translate
  DESTINATION bin/
  )
