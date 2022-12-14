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

void SGBM(const cv::Mat& imageL, const cv::Mat& imageR, cv::Mat& disp) {
    cv::Mat grayImageL, grayImageR;
	cv::cvtColor(imageL, grayImageL, cv::COLOR_BGR2GRAY);
	cv::cvtColor(imageR, grayImageR, cv::COLOR_BGR2GRAY);

    // cv::namedWindow("ImageL", cv::WINDOW_NORMAL);
    // cv::namedWindow("ImageR", cv::WINDOW_NORMAL);
	// cv::imshow("ImageL", grayImageL);
	// cv::imshow("ImageR", grayImageR);
    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(0, 16, 3);
    int numberOfDisparities = ((imageL.size().width / 8) + 15)&-16;
    // std::cout << numberOfDisparities << std::endl;
	int SADWindowSize = 5;
	int sgbmWinSize = SADWindowSize > 0 ? SADWindowSize : 3;
	int cn = imageL.channels();
    std::cout << CV_VERSION << std::endl;

    sgbm->setPreFilterCap(32);
	sgbm->setBlockSize(sgbmWinSize);
	sgbm->setP1(8 * cn * sgbmWinSize * sgbmWinSize);
	sgbm->setP2(32 * cn * sgbmWinSize * sgbmWinSize);
	sgbm->setMinDisparity(0);
	sgbm->setNumDisparities(numberOfDisparities);
	sgbm->setUniquenessRatio(10);
	sgbm->setSpeckleWindowSize(100);
	sgbm->setSpeckleRange(32);
	sgbm->setDisp12MaxDiff(1);

    sgbm->compute(grayImageL, grayImageR, disp);
}

int main(int argc, char const *argv[])
{
    cv::Mat rgbImageL = cv::imread("../images/im0.png", cv::IMREAD_COLOR);
	cv::Mat rgbImageR = cv::imread("../images/im1.png", cv::IMREAD_COLOR);

    cv::Mat disp;
    // SGBM算法
    SGBM(rgbImageL, rgbImageR, disp);

    // 除以16得到真实的视差
    disp.convertTo(disp, CV_32F, 1.0 / 16);
    // 转化为8U类型
    cv::Mat disp8U = cv::Mat(disp.rows, disp.cols, CV_8UC1);
    // 归一化
    cv::normalize(disp, disp8U, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    // 伪彩色图
    cv::Mat colorMap;
    cv::applyColorMap(disp8U, colorMap, cv::COLORMAP_JET);
    cv::imwrite("../images/output/SGBM.png", colorMap);
    // cv::namedWindow("Disparity", cv::WINDOW_NORMAL);
	// cv::imshow("Disparity", colorMap);
    
    cv::waitKey();
    cv::destroyAllWindows();
    return 0;
}
