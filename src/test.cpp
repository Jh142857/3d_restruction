// #include "stdafx.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/types_c.h"
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <opencv2/opencv.hpp>
// #include "cv.h"
// #include <cv.hpp>


using namespace std;
using namespace cv;

const int imageWidth =640;                             //摄像头的分辨率  
const int imageHeight =480;
Size imageSize = Size(imageWidth, imageHeight);

Mat rgbImageL, grayImageL;
Mat rgbImageR, grayImageR;
Mat rectifyImageL, rectifyImageR;

Rect validROIL;//图像校正之后，会对图像进行裁剪，这里的validROI就是指裁剪之后的区域  
Rect validROIR;

Mat mapLx, mapLy, mapRx, mapRy;     //映射表  
Mat Rl, Rr, Pl, Pr, Q;              //校正旋转矩阵R，投影矩阵P 重投影矩阵Q
Mat xyz;              //三维坐标

Point origin;         //鼠标按下的起始点
Rect selection;      //定义矩形选框
bool selectObject = false;    //是否选择对象

int blockSize = 0, uniquenessRatio = 0, numDisparities = 0;
Ptr<StereoBM> bm = StereoBM::create(16, 9);

Mat cameraMatrixL = (Mat_<double>(3, 3) << 705.2421, -0.8178, 304.2868,
	0, 704.9515, 237.4467,
	0, 0, 1);

Mat distCoeffL = (Mat_<double>(5, 1) << 0.0045, 1.1431, -0.0004, -0.0026, -5.1702);

Mat cameraMatrixR = (Mat_<double>(3, 3) << 701.6815, 0.7489, 308.5031,
	0, 701.7459, 258.5190,
	0, 0, 1);

Mat distCoeffR = (Mat_<double>(5, 1) <<0.0008,1.1708, -0.0002, 0.0031,-5.1702);

Mat T = (Mat_<double>(3, 1) <<23.2464, -0.1463, 1.7391);//T平移向量 对应Matlab所得T参数

Mat rec = (Mat_<double>(3, 1) << -0.0040 , 0.1807   , -0.0067);//rec旋转向量，对应matlab om参数

Mat R;

class SGBM
{
private:
	enum mode_view { LEFT, RIGHT };
	mode_view view;	//输出左视差图or右视差图

public:
	SGBM() {};
	SGBM(mode_view _mode_view) :view(_mode_view) {};
	~SGBM() {};
	Mat computersgbm(Mat &L, Mat &R);	//计算SGBM
};

Mat SGBM::computersgbm(Mat &L, Mat &R)
/*SGBM_matching SGBM算法
*@param Mat &left_image :左图像
*@param Mat &right_image:右图像
*/
{
	Mat disp;

	int numberOfDisparities = ((L.size().width / 8) + 15)&-16;
	Ptr<StereoSGBM> sgbm = StereoSGBM::create(0, 16, 3);
	sgbm->setPreFilterCap(32);

	int SADWindowSize = 5;
	int sgbmWinSize = SADWindowSize > 0 ? SADWindowSize : 3;
	sgbm->setBlockSize(sgbmWinSize);
	int cn = L.channels();

	sgbm->setP1(8 * cn*sgbmWinSize*sgbmWinSize);
	sgbm->setP2(32 * cn*sgbmWinSize*sgbmWinSize);
	sgbm->setMinDisparity(0);
	sgbm->setNumDisparities(numberOfDisparities);
	sgbm->setUniquenessRatio(10);
	sgbm->setSpeckleWindowSize(100);
	sgbm->setSpeckleRange(32);
	sgbm->setDisp12MaxDiff(1);


	Mat left_gray, right_gray;
	cvtColor(L, left_gray, cv::COLOR_BGR2GRAY);
	cvtColor(R, right_gray, cv::COLOR_BGR2GRAY);

	view = LEFT;
	if (view == LEFT)	//计算左视差图
	{
		sgbm->compute(left_gray, right_gray, disp);

		disp.convertTo(disp, CV_32F, 1.0 / 16);			//除以16得到真实视差值

		Mat disp8U = Mat(disp.rows, disp.cols, CV_8UC1);
		normalize(disp, disp8U, 0, 255, NORM_MINMAX, CV_8UC1);
		imwrite("SGBM.jpg", disp8U);

		return disp8U;
	}
	else if (view == RIGHT)	//计算右视差图
	{
		sgbm->setMinDisparity(-numberOfDisparities);
		sgbm->setNumDisparities(numberOfDisparities);
		sgbm->compute(left_gray, right_gray, disp);

		disp.convertTo(disp, CV_32F, 1.0 / 16);			//除以16得到真实视差值

		Mat disp8U = Mat(disp.rows, disp.cols, CV_8UC1);
		normalize(disp, disp8U, 0, 255, NORM_MINMAX, CV_8UC1);
		imwrite("SGBM.jpg", disp8U);

		return disp8U;
	}
	else
	{
		return Mat();
	}
}


int main()
{	
	/*
	立体校正
	*/
	Rodrigues(rec, R); //Rodrigues变换
	stereoRectify(cameraMatrixL, distCoeffL, cameraMatrixR, distCoeffR, imageSize, R, T, Rl, Rr, Pl, Pr, Q, CALIB_ZERO_DISPARITY,
		0, imageSize, &validROIL, &validROIR);
	initUndistortRectifyMap(cameraMatrixL, distCoeffL, Rl, Pr, imageSize, CV_32FC1, mapLx, mapLy);
	initUndistortRectifyMap(cameraMatrixR, distCoeffR, Rr, Pr, imageSize, CV_32FC1, mapRx, mapRy);

	/*
	读取图片
	*/
	rgbImageL = imread("1-left.jpg", cv::IMREAD_COLOR);
	cvtColor(rgbImageL, grayImageL, cv::COLOR_BGR2GRAY);
	rgbImageR = imread("1-right.jpg", cv::IMREAD_COLOR);
	cvtColor(rgbImageR, grayImageR, cv::COLOR_BGR2GRAY);

	imshow("ImageL Before Rectify", grayImageL);
	imshow("ImageR Before Rectify", grayImageR);

	/*
	经过remap之后，左右相机的图像已经共面并且行对准了
	*/
	remap(grayImageL, rectifyImageL, mapLx, mapLy, INTER_LINEAR);
	remap(grayImageR, rectifyImageR, mapRx, mapRy, INTER_LINEAR);

	/*
	把校正结果显示出来
	*/
	Mat rgbRectifyImageL, rgbRectifyImageR;
	cvtColor(rectifyImageL, rgbRectifyImageL, cv::COLOR_GRAY2BGR);  //伪彩色图
	cvtColor(rectifyImageR, rgbRectifyImageR, cv::COLOR_GRAY2BGR);

	//显示在同一张图上
	Mat canvas;
	double sf;
	int w, h;
	sf = 600. / MAX(imageSize.width, imageSize.height);
	w = cvRound(imageSize.width * sf);
	h = cvRound(imageSize.height * sf);
	canvas.create(h, w * 2, CV_8UC3);   //注意通道

										//左图像画到画布上
	Mat canvasPart = canvas(Rect(w * 0, 0, w, h));                                //得到画布的一部分  
	resize(rgbRectifyImageL, canvasPart, canvasPart.size(), 0, 0, INTER_AREA);     //把图像缩放到跟canvasPart一样大小  
	Rect vroiL(cvRound(validROIL.x*sf), cvRound(validROIL.y*sf),                //获得被截取的区域    
		cvRound(validROIL.width*sf), cvRound(validROIL.height*sf));
	//rectangle(canvasPart, vroiL, Scalar(0, 0, 255), 3, 8);                      //画上一个矩形  
	cout << "Painted ImageL" << endl;

	//右图像画到画布上
	canvasPart = canvas(Rect(w, 0, w, h));                                      //获得画布的另一部分  
	resize(rgbRectifyImageR, canvasPart, canvasPart.size(), 0, 0, INTER_LINEAR);
	Rect vroiR(cvRound(validROIR.x * sf), cvRound(validROIR.y*sf),
		cvRound(validROIR.width * sf), cvRound(validROIR.height * sf));
	//rectangle(canvasPart, vroiR, Scalar(0, 0, 255), 3, 8);
	cout << "Painted ImageR" << endl;

	//画上对应的线条
	for (int i = 0; i < canvas.rows; i += 16)
		line(canvas, Point(0, i), Point(canvas.cols, i), Scalar(0, 255, 0), 1, 8);
	imshow("rectified", canvas);

	Mat left = rgbRectifyImageL;
	Mat right = rgbRectifyImageR;

	//-------图像显示-----------
	namedWindow("leftimag");
	imshow("leftimag", left);

	namedWindow("rightimag");
	imshow("rightimag", right);
	//--------由SAD求取视差图-----
	Mat Disparity;

	SGBM mySGBM;
	Disparity = mySGBM.computersgbm(left, right);

	//-------结果显示------
	namedWindow("Disparity");
	imshow("Disparity", Disparity);
	imwrite("disparity.jpg", Disparity);
	


	waitKey(0);
	return 0;
}
