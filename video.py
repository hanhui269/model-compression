#视频的读取
import cv2
import numpy as np

import math

# vc=cv2.VideoCapture('./12.mp4')
vc=cv2.VideoCapture('video.avi')


# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('test.avi', fourcc, 20.0, (640, 480))


if vc.isOpened():
  open,frame=vc.read()
else:
  open = False


while open:
    ret,frame=vc.read()
    if frame is None:
        break
    if ret == True:
        # gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)#将彩色视频转化为灰色视频
        # gray_filter=cv2.medianBlur(gray,5)
        # sobelx8u = cv2.Sobel(gray_filter, cv2.CV_8U, 1, 0, ksize=3)
        # lines = cv2.HoughLinesP(edges, 1, cv2.cv.CV_PI / 180, minLINELENGTH, 0)
        # frame = cv2.resize(frame,(640,384))
        frame = cv2.medianBlur(frame, 5)  # 均值滤波
        # 限制对比度的自适应阈值均衡化
        # 创建CLAHE对象


        # 使用全局直方图均衡化
        # equa = cv2.equalizeHist(frame)


        cv2.imshow("pic",frame)
        cv2.waitKey(0)



        if cv2.waitKey(10)&0xFF==27:#括号中数字越大，视频播放速度越慢。0xFF==27表示按ESC后退出视频播放
           break


vc.release()
cv2.destroyAllWindows()