cmake_minimum_required(VERSION 3.1)
project(fcnn)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rd/eigen/include)


file(GLOB fcnn_src src/fcnn.cc)
add_library(${PROJECT_NAME} SHARED ${fcnn_src})

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/3rd/googletest)


file(GLOB_RECURSE test_src "test/*.cc")
add_executable(test_main ${test_src})
target_link_libraries(test_main PUBLIC gtest gtest_main)