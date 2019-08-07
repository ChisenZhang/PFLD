# !/usr/bin/python
# encoding:utf-8

'''
训练集数据统计
'''

import sys
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

import cv2

sys.path.append('./dataLoader')
sys.path.append('./dataLoader/WIDE_FACE')

from WIDE_FACE.wider_voc import VOCDetection, AnnotationTransform
from dataLoader import DataService

data_train_dir = '/home/wei.ma/face_detection/FaceBoxes.PyTorch/data/WIDER_FACE/'
data_test_dir = '/home/wei.ma/face_detection/FaceBoxes.PyTorch/data/FDDB'

if __name__ == '__main__':
    IM_S = 256
    BATCH_SIZE = 32
    train_data = VOCDetection(data_train_dir, target_transform=AnnotationTransform())
    print('train_wideFaceNum:', len(train_data))

    data_loader = DataService(train_data, BATCH_SIZE, 32, workers=3)
    data_loader.start()
    epoch_Steps = len(train_data)//BATCH_SIZE
    step = 0
    maxWidth = {}
    ratio = {}

    for k in range(2):
        print('epoch:', k)

        flag = True
        while True:
            imgs, lbls = data_loader.pop()
            print('step:', step)
            for i in range(BATCH_SIZE):
                img = imgs[i]
                r = 256./max(img.shape[0], img.shape[1])
                for k in range(len(lbls[i])):
                    box = lbls[i][k]
                    x = box[0]*r
                    y = box[1]*r
                    w = box[2]*r - x
                    h = box[3]*r - y
                    tmpMWH = int(max(w, h))
                    tmpRatio = round(w/h, 3)
                    if maxWidth.__contains__(tmpMWH):
                        maxWidth[tmpMWH] += 1
                    else:
                        maxWidth[tmpMWH] = 1
                    if ratio.__contains__(tmpRatio):
                        ratio[tmpRatio] += 1
                    else:
                        ratio[tmpRatio] = 1
            if step != 0 and step % epoch_Steps == 0:
                step += 1
                break

            step += 1
        break
    print('dump data to Txt')
    fmwh = open('fmwh.txt', 'w', encoding='utf-8')
    for key in maxWidth.keys():
        fmwh.write(str(key)+'\t'+str(maxWidth[key])+'\n')
    fmwh.close()
    fratio = open('fratio.txt', 'w', encoding='utf-8')
    for key in ratio.keys():
        fratio.write(str(key) + '\t' + str(ratio[key]) + '\n')
    fratio.close()
    data_loader.stop()
    print('finish')
