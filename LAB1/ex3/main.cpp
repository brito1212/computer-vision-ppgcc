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

	int x = 0;
	int y = 0;

	std::cout << "Intervalos validos: x [0, "
				  << grayImage.cols - 1 << "], y [0, " << grayImage.rows - 1 << "]" << '\n';
	std::cout << "Digite as coordenadas do pixel (x y): ";
	if (!(std::cin >> x >> y)) {
		std::cerr << "Entrada invalida. Digite dois inteiros: x y" << '\n';
		return 1;
	}

	if (x < 0 || x >= grayImage.cols || y < 0 || y >= grayImage.rows) {
		std::cerr << "Coordenadas fora da imagem. Intervalos validos: x [0, "
				  << grayImage.cols - 1 << "], y [0, " << grayImage.rows - 1 << "]" << '\n';
		return 1;
	}

	const int intensity = static_cast<int>(grayImage.at<uchar>(y, x));
	std::cout << "Intensidade em (" << x << ", " << y << "): " << intensity << '\n';

	cv::imshow("Escala de cinza", grayImage);
	cv::waitKey(0);


	return 0;
}