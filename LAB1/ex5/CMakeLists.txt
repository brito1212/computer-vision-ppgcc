add_executable(ex5 main.cpp)
target_link_libraries(ex5 PRIVATE ${OpenCV_LIBS})
target_compile_definitions(ex5 PRIVATE LAB1_IMAGE_PATH="${CMAKE_SOURCE_DIR}/imagem.jpg")