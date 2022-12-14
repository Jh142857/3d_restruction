# 3D_Restrcution
## Intruction
该Repo采用opencv4.6.0实现双目立体匹配，使用的方法如下
- BM：`cv::StereoBM`
- SGBM：`cv::StereoSGBM`
- Quasi：`cv::stereo::QuasiDenseStereo`

## Run
- 运行环境：ubuntu20.04
- 首先确保已经安装了opencv和opencv-contrib，版本最好是4.6.0
- 然后编译运行
```bash
./run.sh
```
## Report
见根目录[计算机视觉-三维重建作业.pdf](计算机视觉-三维重建作业.pdf)