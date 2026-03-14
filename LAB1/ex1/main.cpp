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

	std::cout << "Largura: " << image.cols << '\n';
	std::cout << "Altura: " << image.rows << '\n';
	std::cout << "Canais: " << image.channels() << '\n';

	cv::imshow("Imagem", image);
	cv::waitKey(0);

	return 0;
}