add_executable(ex1 main.cpp)
target_link_libraries(ex1 PRIVATE ${OpenCV_LIBS})
target_compile_definitions(ex1 PRIVATE LAB1_IMAGE_PATH="${CMAKE_SOURCE_DIR}/imagem.jpg")