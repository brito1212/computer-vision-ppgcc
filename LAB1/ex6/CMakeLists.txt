add_executable(ex6 main.cpp)
target_link_libraries(ex6 PRIVATE ${OpenCV_LIBS})
target_compile_definitions(ex6 PRIVATE LAB1_IMAGE_PATH="${CMAKE_SOURCE_DIR}/imagem.jpg")