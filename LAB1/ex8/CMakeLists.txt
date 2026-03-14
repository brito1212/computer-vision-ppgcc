add_executable(ex8 main.cpp)
target_link_libraries(ex8 PRIVATE ${OpenCV_LIBS})
target_compile_definitions(ex8 PRIVATE LAB1_IMAGE_PATH="${CMAKE_SOURCE_DIR}/imagem.jpg")