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

	const int halfWidth = image.cols / 2;
	const int halfHeight = image.rows / 2;

	cv::Mat reducedImage;
	cv::Mat restoredImage;

	cv::resize(image, reducedImage, cv::Size(halfWidth, halfHeight), 0, 0, cv::INTER_AREA);
	cv::resize(reducedImage, restoredImage, image.size(), 0, 0, cv::INTER_LINEAR);

	cv::imshow("Original", image);
	cv::imshow("Reduzida", reducedImage);
	cv::imshow("Redimensionada para resolucao original", restoredImage);
	cv::waitKey(0);

	const std::filesystem::path basePath = std::filesystem::path(__FILE__).parent_path();
	cv::imwrite((basePath / "resultado_original.jpg").string(), image);
	cv::imwrite((basePath / "resultado_reduzida.jpg").string(), reducedImage);
	cv::imwrite((basePath / "resultado_restaurada.jpg").string(), restoredImage);
	std::cout << "Imagens salvas em: " << basePath << '\n';

	return 0;
}