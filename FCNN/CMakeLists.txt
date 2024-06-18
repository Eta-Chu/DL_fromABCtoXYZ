set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-std=c++17 -lstdc++ -pipe -O3 -fPIC")
cmake_minimum_required(VERSION 3.1)
project(fcnn)
set(CMAKE_CXX_STANDARD 14)

# 查找OpenMP库
find_package(OpenMP REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rd/eigen/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

file(GLOB fcnn_src src/*.cc)
add_library(nn SHARED ${fcnn_src})
target_link_libraries(nn PUBLIC OpenMP::OpenMP_CXX)

#add_subdirectory(${CMAKE_CRRENT_SOURCE_DIR}/3rd/googletest)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/3rd/glog)

file(GLOB test_src "test/*.cpp")
add_executable(test ${test_src} ${fcnn_src})
target_link_libraries(test PUBLIC nn OpenMP::OpenMP_CXX glog)
#target_link_libraries(test_main PUBLIC gtest gtest_main)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/build)
file(GLOB MAIN_SRC main/main.cc)
add_executable(main ${MAIN_SRC})
target_link_libraries(main PUBLIC nn glog OpenMP::OpenMP_CXX)
