# !/usr/bin/python
# encoding:utf-8

'''
解析Vedio to image
'''

import cv2
import uuid
import os


def parsingVedio(vedioPath, imgStorePath):
    cap = cv2.VideoCapture(vedioPath)
    print('vedio Open:', cap.isOpened())
    frame_count = 1
    wid = int(cap.get(3))
    hei = int(cap.get(4))
    framerate = int(cap.get(5))
    framenum = int(cap.get(7))
    print('vedio info: w:%d, h:%d, frameRate:%d, frameNum:%d'%(wid, hei, framerate, framenum))
    success = True
    nameHead = str(uuid.uuid4()).replace('-', '')
    while success:
        success, frame = cap.read()

        if not success:
            break

        # 全黑
        if frame.mean() < 50:
            frame_count += 1
            continue

        print('save frame:', frame_count)
        params = []
        # params.append(cv2.CV_IMWRITE_PXM_BINARY)
        params.append(1)
        cv2.imwrite(os.path.join(imgStorePath, nameHead+'_'+str(frame_count)+'.jpg'), frame)
        frame_count += 1
    cap.release()


if __name__ == '__main__':
    vedioPath = r'C:\Users\17ZY-HPYKFD2\Downloads\dFServer\u_230524394_45_2019-07-14_5d2ac3c56816f8571c1198c0.mp4'
    imStorePath = './tmpImg'
    if not os.path.exists(imStorePath):
        os.mkdir(imStorePath)
    parsingVedio(vedioPath, imStorePath)
