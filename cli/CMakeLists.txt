set(Boost_USE_STATIC_LIBS ON)
find_package(Boost COMPONENTS program_options)

add_executable(translate
  translate.cc
  )
target_link_libraries(translate PRIVATE
  ${PROJECT_NAME}
  ${Boost_LIBRARIES}
  )

install(
  TARGETS translate
  DESTINATION bin/
  )
