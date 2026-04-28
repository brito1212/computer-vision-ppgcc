cmake_minimum_required(VERSION 3.10)
project(hough_pratos)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(hough_pratos hough_pratos.cpp)

target_link_libraries(hough_pratos ${OpenCV_LIBS})
