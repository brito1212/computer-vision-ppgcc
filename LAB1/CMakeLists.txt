cmake_minimum_required(VERSION 3.10)

project(LAB1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)

add_subdirectory(ex1)
add_subdirectory(ex2)
add_subdirectory(ex3)
add_subdirectory(ex4)
add_subdirectory(ex5)
add_subdirectory(ex6)
add_subdirectory(ex7)
add_subdirectory(ex8)