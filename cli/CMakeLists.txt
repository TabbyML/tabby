find_package(Boost COMPONENTS program_options)

add_executable(translate
  translate.cc
  )
target_include_directories(translate PUBLIC
  ${Boost_INCLUDE_DIRS}
  )
target_link_libraries(translate
  ${PROJECT_NAME}
  ${Boost_LIBRARIES}
  )

install(
  TARGETS translate
  DESTINATION bin/
  )
