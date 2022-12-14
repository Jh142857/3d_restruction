import cv2


image1 = cv2.imread("./images/im2.png", flags=3)
image2 = cv2.imread("./images/im3.png", flags=3)
bm = cv2.StereoBM_create(numDisparities=16, blockSize=9)
numberOfDisparities = ((1920 // 8) + 15) & -16  # 640对应是分辨率的宽
 
bm.setPreFilterCap(31)
bm.setBlockSize(15)
bm.setMinDisparity(0)
bm.setNumDisparities(numberOfDisparities)
bm.setTextureThreshold(10)
bm.setUniquenessRatio(15)
bm.setSpeckleWindowSize(100)
bm.setSpeckleRange(32)
bm.setDisp12MaxDiff(1)

window_size = 3
sgbm = cv2.StereoSGBM_create(
    minDisparity=-1,
    numDisparities=5*16,  # max_disp has to be dividable by 16 f. E. HH 192, 256
    blockSize = window_size,
    P1=8 * 3 * window_size,
    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
    P2=32 * 3 * window_size,
    disp12MaxDiff=12,
    uniquenessRatio=10,
    speckleWindowSize=50,
    speckleRange=32,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)
 
disparity = sgbm.compute(image1, image2)
cv2.namedWindow("disparity", cv2.WINDOW_NORMAL)
cv2.imshow("disparity", disparity)
cv2.waitKey(0)
cv2.destroyAllWindows()