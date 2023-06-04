cmake_minimum_required(VERSION 3.10)

project(tabby-coreml)

add_executable(tabby-coreml
  lib.mm
)

target_link_libraries(tabby-coreml stdc++ "-framework Foundation" "-framework CoreML" objc)

set_target_properties(tabby-coreml PROPERTIES
  CXX_STANDARD 17
  OBJCXX_STANDARD 17
)

if(APPLE)
  set(CMAKE_OSX_DEPLOYMENT_TARGET 10.13)
endif()
