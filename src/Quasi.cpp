#include <iostream>
#include <algorithm>
// 核心库，包含一些cv常量
#include <opencv2/core.hpp>
// 标定重建工具箱
#include <opencv2/calib3d.hpp>
// 图像处理库
#include <opencv2/imgproc.hpp>
// 读写图片相关库
#include <opencv2/imgcodecs.hpp>
// 图像GUI
#include <opencv2/highgui.hpp>

#include <opencv2/stereo/quasi_dense_stereo.hpp>

void quasi(const cv::Mat& imageL, const cv::Mat& imageR, cv::Mat& disp) {
    cv::Mat grayImageL, grayImageR;
	cv::cvtColor(imageL, grayImageL, cv::COLOR_BGR2GRAY);
	cv::cvtColor(imageR, grayImageR, cv::COLOR_BGR2GRAY);

    // cv::namedWindow("ImageL", cv::WINDOW_NORMAL);
    // cv::namedWindow("ImageR", cv::WINDOW_NORMAL);
	// cv::imshow("ImageL", grayImageL);
	// cv::imshow("ImageR", grayImageR);
    cv::Ptr<cv::stereo::QuasiDenseStereo> quasi = cv::stereo::QuasiDenseStereo::create(grayImageL.size());
    // 处理图片
    quasi->process(grayImageL, grayImageR);
    // // 得到视差图
    // uint8_t displvl = 80;
    // cv::Mat disp;
    disp = quasi->getDisparity();
}

int main(int argc, char const *argv[])
{
    cv::Mat rgbImageL = cv::imread("../images/im0.png", cv::IMREAD_COLOR);
	cv::Mat rgbImageR = cv::imread("../images/im1.png", cv::IMREAD_COLOR);

    cv::Mat disp;
    // SGBM算法
    // SGBM(rgbImageL, rgbImageR, disp);
    quasi(rgbImageL, rgbImageR, disp);

    // 转化为8U类型
    disp.convertTo(disp, CV_8U);
    
    // 伪彩色图
    cv::Mat colorMap;
    cv::applyColorMap(disp, colorMap, cv::COLORMAP_JET);
    cv::imwrite("../images/output/Quasi.png", colorMap);
    // cv::namedWindow("Disparity", cv::WINDOW_NORMAL);
	// cv::imshow("Disparity", colorMap);
    
    cv::waitKey();
    cv::destroyAllWindows();
    return 0;
}
