cmake_minimum_required(VERSION 3.10)
project(LiDAR)

set(CMAKE_CXX_STANDARD 17)

find_package(PCL 1.10 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(main src/main.cpp)
target_link_libraries(main ${PCL_LIBRARIES})
