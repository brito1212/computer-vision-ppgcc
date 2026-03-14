add_executable(ex4 main.cpp)
target_link_libraries(ex4 PRIVATE ${OpenCV_LIBS})
target_compile_definitions(ex4 PRIVATE LAB1_IMAGE_PATH="${CMAKE_SOURCE_DIR}/imagem.jpg")