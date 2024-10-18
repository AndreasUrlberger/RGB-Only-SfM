#include <opencv2/opencv.hpp>

int main(int argc, char** argv)
{
    cv::Mat img = cv::imread(argv[1], -1);
    if (img.empty()) return -1;
    std::cout << "Test image size: " << img.size() << std::endl;
    return 0;
}