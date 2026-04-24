#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>

cv::Mat quantizeGray4Levels(const cv::Mat& grayImage) {
	const int step = 256 / 4;
	cv::Mat lut(1, 256, CV_8U);

	for (int i = 0; i < 256; ++i) {
		lut.at<uchar>(i) = static_cast<uchar>((i / step) * step);
	}

	cv::Mat quantized;
	cv::LUT(grayImage, lut, quantized);
	return quantized;
}

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
	cv::resize(image, reducedImage, cv::Size(halfWidth, halfHeight), 0, 0, cv::INTER_AREA);

	cv::Mat grayImage;
	cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
	cv::Mat gray4Levels = quantizeGray4Levels(grayImage);

	cv::Mat reducedGray;
	cv::cvtColor(reducedImage, reducedGray, cv::COLOR_BGR2GRAY);
	cv::Mat reducedGray4Levels = quantizeGray4Levels(reducedGray);

	cv::imshow("Original", image);
	cv::imshow("1) Resolucao reduzida", reducedImage);
	cv::imshow("2) Cinza com 4 niveis", gray4Levels);
	cv::imshow("3) Reduzida + 4 niveis de cinza", reducedGray4Levels);
	cv::waitKey(0);

	const std::filesystem::path basePath = std::filesystem::path(__FILE__).parent_path();
	cv::imwrite((basePath / "resultado_original.jpg").string(), image);
	cv::imwrite((basePath / "resultado_reduzida.jpg").string(), reducedImage);
	cv::imwrite((basePath / "resultado_cinza4niveis.jpg").string(), gray4Levels);
	cv::imwrite((basePath / "resultado_reduzida_cinza4niveis.jpg").string(), reducedGray4Levels);
	std::cout << "Imagens salvas em: " << basePath << '\n';

	return 0;
}