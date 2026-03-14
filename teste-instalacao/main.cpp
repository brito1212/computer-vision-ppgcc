#include <iostream>
#include <opencv2/opencv.hpp>
    int main() {
    cv::Mat img(300, 400, CV_8UC3, cv::Scalar(0, 255, 0));
    cv::imshow("Teste OpenCV", img);
    cv::waitKey(0);
    std::cout << "OpenCV funcionando corretamente.\n";
    return 0;
}