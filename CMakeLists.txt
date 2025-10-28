cmake_minimum_required(VERSION 3.10)
project(LiDAR)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(PCL 1.10 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

find_package(ZeroMQ REQUIRED)
find_package(CPPZMQ_INCLUDE_DIR REQUIRED)
include_directories(${CPPZMQ_INCLUDE_DIR})

add_subdirectory(as-serializers)
target_link_libraries(main
       as-serializers
   )  


add_executable(main src/main.cpp)
target_link_libraries(main PRIVATE
    ${PCL_LIBRARIES}
    zmq
)


