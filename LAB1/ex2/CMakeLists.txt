add_executable(ex2 main.cpp)
target_link_libraries(ex2 PRIVATE ${OpenCV_LIBS})
target_compile_definitions(ex2 PRIVATE LAB1_IMAGE_PATH="${CMAKE_SOURCE_DIR}/imagem.jpg")