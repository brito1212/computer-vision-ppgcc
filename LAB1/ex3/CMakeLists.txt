add_executable(ex3 main.cpp)
target_link_libraries(ex3 PRIVATE ${OpenCV_LIBS})
target_compile_definitions(ex3 PRIVATE LAB1_IMAGE_PATH="${CMAKE_SOURCE_DIR}/imagem.jpg")