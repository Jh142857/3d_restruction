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

int preFilterCap = 13;
int blockSize = 8;
int minDisparity = 16; // 最小视差，默认值为0
int numDisparities = 240; // 最大视差值与最小视差值之差，必须是16的整数倍
int textureThreshold = 1000; // 低纹理区域判断阈值，保证有足够的纹理以克服噪声 
int uniquenessRatio = 1; // 视差唯一性百分比
int speckleWindowSize = 3; // 检查视差连通区域变化度的窗口大小, 值为0时取消 speckle 检查  
int speckleRange = 3; // 视差变化阈值，当窗口内视差变化大于阈值时，该窗口内的视差清零  
int disp12MaxDiff = 200;                                                                   

void getDisparity(const cv::Mat& imageLeft, const cv::Mat& imageRight, cv::Mat& disparity) {
    cv::Ptr<cv::StereoBM> bm = cv::StereoBM::create();
    bm->setPreFilterCap(preFilterCap); // 预处理滤波器的判断阈值
    bm->setBlockSize(2 * blockSize - 1); // SAD窗口大小
    // bm->setROI1();
    bm->setMinDisparity(minDisparity); // 最小视差，默认值为0
    bm->setNumDisparities(numDisparities); // 最大视差值与最小视差值之差，必须是16的整数倍
    bm->setTextureThreshold(textureThreshold); // 低纹理区域判断阈值，保证有足够的纹理以克服噪声 
    bm->setUniquenessRatio(uniquenessRatio); // 视差唯一性百分比
    bm->setSpeckleWindowSize(speckleWindowSize); // 检查视差连通区域变化度的窗口大小, 值为0时取消 speckle 检查  
    bm->setSpeckleRange(speckleRange); // 视差变化阈值，当窗口内视差变化大于阈值时，该窗口内的视差清零  
    bm->setDisp12MaxDiff(disp12MaxDiff); //左视差图（直接计算得出）和右视差图（通过cvValidateDisparity计算得出）,之间的最大容许差异，默认为-1 

    bm->compute(imageLeft, imageRight, disparity);
}

static void onChange(int pos, void* data) {
    std::vector<cv::Mat> images = *((std::vector<cv::Mat>*)data);
    cv::Mat disparity;
    getDisparity(images[0], images[1], disparity);
    cv::imshow("disparity", disparity);
}

void colorMap(const cv::Mat& disparity, cv::Mat& colorImage){
    // int min = *std::min_element(disparity.begin<int>(), disparity.end<int>());
    // int max = *std::max_element(disparity.begin<int>(), disparity.end<int>());
    // cv::Mat images_re = disparity.reshape(1);

    cv::Mat disparityNorm = disparity.clone();
    double min, max;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(disparityNorm, &min, &max, &minLoc, &maxLoc);
    disparityNorm -= min;
    disparityNorm = disparityNorm / (max - min);
    disparityNorm *= 255;
    // disparityNorm = 255 - disparityNorm;
    // std::cout << min << " " << max << std::endl;
    // std::cout << disparityNorm << std::endl;
    
    cv::Mat disparityScale;
    cv::convertScaleAbs(disparityNorm, disparityScale);
    cv::applyColorMap(disparityScale, colorImage, cv::COLORMAP_JET);
    // std::cout << colorImage << std::endl;
}

int main(int argc, char const *argv[])
{
    // 采用灰度图模式读取原始图
    int colorMode = cv::IMREAD_GRAYSCALE;
    cv::Mat imageLeft = cv::imread("../images/im0.png", colorMode);
    cv::Mat imageRight = cv::imread("../images/im1.png", colorMode);
    // std::cout << imageLeft.type() << std::endl;
    cv::Mat disparity;

    getDisparity(imageLeft, imageRight, disparity);
    // std::cout << disparity << std::endl;

    // cv::Mat disparityGray, disparityRGB;
    // // 转换图像的存储格式（深度）
    // disparity.convertTo(disparityGray, CV_8U);
    // // 转换为rgb图片
    // cv::cvtColor(disparityGray, disparityRGB, cv::COLOR_GRAY2RGB);

    // 伪彩色图增强
    // cv::Mat colorDisparity;
    // colorMap(disparity, colorDisparity);

    // cv::imwrite("../images/out.jpg", disparity);
    
    cv::namedWindow("disparity", cv::WINDOW_NORMAL);
    cv::resizeWindow("disparity", 960, 540);
    cv::imshow("disparity", disparity);

    // 创建进度条
    std::vector<cv::Mat> images;
    images.push_back(imageLeft);
    images.push_back(imageRight);
    cv::namedWindow("Track Bar");
    cv::createTrackbar("preFilterCap", "Track Bar", &preFilterCap, 63, onChange, &images);
    cv::createTrackbar("blockSize", "Track Bar", &blockSize, 128, onChange, &images);
    cv::createTrackbar("minDisparity", "Track Bar", &minDisparity, 100, onChange, &images);
    cv::createTrackbar("numDisparities", "Track Bar", &numDisparities, 100, onChange, &images);
    cv::createTrackbar("textureThreshold", "Track Bar", &textureThreshold, 10000, onChange, &images);
    cv::createTrackbar("speckleWindowSize", "Track Bar", &speckleWindowSize, 100, onChange, &images);
    cv::createTrackbar("speckleRange", "Track Bar", &speckleRange, 100, onChange, &images);
    cv::createTrackbar("uniquenessRatio", "Track Bar", &speckleRange, 100, onChange, &images);
    cv::createTrackbar("disp12MaxDiff", "Track Bar", &disp12MaxDiff, 1000, onChange, &images);
    
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}
