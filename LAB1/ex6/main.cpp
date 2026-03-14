#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>


cv::Mat quantizeGrayLevels(const cv::Mat& grayImage, int levels) {
	const int step = 256 / levels;

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

	cv::Mat grayImage;

	cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

	cv::Mat gray128;
	cv::Mat gray64;
	cv::Mat gray16;
	cv::Mat gray4;

	gray128 = quantizeGrayLevels(grayImage, 128);
	gray64 = quantizeGrayLevels(grayImage, 64);
	gray16 = quantizeGrayLevels(grayImage, 16);
	gray4 = quantizeGrayLevels(grayImage, 4);

	cv::imshow("Cinza 256 niveis", grayImage);
	cv::imshow("Cinza 128 niveis", gray128);
	cv::imshow("Cinza 64 niveis", gray64);
	cv::imshow("Cinza 16 niveis", gray16);
	cv::imshow("Cinza 4 niveis", gray4);
	cv::waitKey(0);
	
	
	return 0;
}