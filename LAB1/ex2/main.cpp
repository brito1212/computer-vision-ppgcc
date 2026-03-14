#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>

int main() {
	const std::filesystem::path imagePath = LAB1_IMAGE_PATH;

	cv::Mat image = cv::imread(imagePath.string(), cv::IMREAD_UNCHANGED);

	if (image.empty()) {
		std::cerr << "Erro ao carregar a imagem: " << imagePath << '\n';
		return 1;
	}

	cv::Mat grayImage;

	cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

	cv::imshow("Original", image);
	cv::imshow("Escala de cinza", grayImage);
	cv::waitKey(0);


	return 0;
}