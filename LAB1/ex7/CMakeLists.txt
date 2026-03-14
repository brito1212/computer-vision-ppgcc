add_executable(ex7 main.cpp)
target_link_libraries(ex7 PRIVATE ${OpenCV_LIBS})
target_compile_definitions(ex7 PRIVATE LAB1_IMAGE_PATH="${CMAKE_SOURCE_DIR}/imagem.jpg")