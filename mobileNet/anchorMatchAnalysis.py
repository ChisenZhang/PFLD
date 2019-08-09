# !/usr/bin/python
# encoding:utf-8

'''
anchor match 分析
'''

import sys
from anchors import Anchors
import cv2
import numpy as np
from lossComput import compute_iou_np, non_max_suppression
import os, uuid
import random

sys.path.append('./dataLoader')
sys.path.append('./dataLoader/WIDE_FACE')

from WIDE_FACE.data_augment import preproc
from WIDE_FACE.wider_voc import VOCDetection, AnnotationTransform
from dataLoader import DataService

data_train_dir = '/home/wei.ma/face_detection/FaceBoxes.PyTorch/data/WIDER_FACE/'
data_test_dir = '/home/wei.ma/face_detection/FaceBoxes.PyTorch/data/FDDB'

def drawBoxes(image, boxes, colorRand=False):
    color = (0, 255, 0)
    for box in boxes:
        if colorRand:
            R = random.randint(0, 255)
            G = random.randint(0, 250)
            B = random.randint(0, 255)
            color = (B, G, R)
        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 3)
        cv2.putText(image, str(int(box[2] - box[0])), (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 255), 2, cv2.LINE_AA)
    # cv2.imwrite(storePath, image)

def anchorFillter_test(anchors, gBoxes, minThresh=0.3, maxThresh=0.7):
    gBoxes = np.array(gBoxes)
    ious = compute_iou_np(anchors, gBoxes)
    # 每个box的最大IOU及相应box 的 index
    # max_iou, max_iou_ids = np.max(ious, axis=0), np.argmax(ious, axis=0)
    # 每个anchor的最大IOU及相应anchor的index
    max_obj_iou, max_obj_iou_ids = np.max(ious, axis=1), np.argmax(ious, axis=1)

    targets = np.ones((anchors.shape[0], 1), dtype=np.int32)*-1
    locs = np.zeros((anchors.shape[0], 4), dtype=np.int32)

    max_obj_iou = np.squeeze(max_obj_iou)
    max_obj_iou_ids = np.squeeze(max_obj_iou_ids)

    targets[max_obj_iou < minThresh] = 0
    # targets[max_obj_iou >= maxThresh] = 1
    max_obj_iou_ids[max_obj_iou < maxThresh] = -1
    pos_inds = max_obj_iou_ids[max_obj_iou_ids >= 0]
    # pos_inds = np.squeeze(pos_inds)
    targets[max_obj_iou_ids >= 0] = 1
    print('match num:', gBoxes.shape, sum(targets == 1), sum(targets == 0))
    if pos_inds.size > 0:
        assertBoxes = gBoxes[pos_inds]
        assertAnchors = anchors[max_obj_iou_ids >= 0, :]

        aC = np.array(assertAnchors[:, 2:] + assertAnchors[:, :2]).astype(np.float32)/2
        aWH = np.array(assertAnchors[:, 2:] - assertAnchors[:, :2]).astype(np.float32)

        bC = np.array(assertBoxes[:, 2:] + assertBoxes[:, :2]).astype(np.float32)/2
        bWH = np.array(assertBoxes[:, 2:] - assertBoxes[:, :2]).astype(np.float32)

        cxy = (bC - aC) / aWH
        wh = np.log(bWH / aWH)

        out = np.concatenate((cxy*10., wh*5.), axis=-1)

        locs[max_obj_iou_ids >= 0] = out

    return locs, np.squeeze(targets), pos_inds


def decode_test(anchor_boxes, locs, nms_thresh=0.3):
    # NOTE: confs is a N x 2 matrix
    # global SCALE_FACTOR

    centers_a = np.array(anchor_boxes[:, 2:] + anchor_boxes[:, :2]) / 2
    w_h_a = np.array(anchor_boxes[:, 2:] - anchor_boxes[:, :2])

    cxcy_in = (np.array(locs[:, :2] + locs[:, 2:])/2)/10.
    wh_in = np.array(locs[:, 2:] - locs[:, :2])/5.

    wh = np.exp(wh_in) * w_h_a
    cxcy = cxcy_in * w_h_a + centers_a

    boxes_out = np.concatenate([cxcy - wh / 2, cxcy + wh / 2], axis=-1)
    keep = non_max_suppression(boxes_out, nms_thresh)
    boxes_out = boxes_out[keep]
    return boxes_out

def decode_batch_test(anchors, locs):
    out_boxes = []
    for i in range(len(locs)):
        b = decode_test(anchors, np.squeeze(locs[i]))
        out_boxes.append(b)
    return out_boxes

if __name__ == '__main__':
    IM_S = 256
    BATCH_SIZE = 32
    tesEnDe = True

    train_data = VOCDetection(data_train_dir, preproc(IM_S, 0, 1/255.), target_transform=AnnotationTransform())
    print('train_wideFaceNum:', len(train_data))

    # NOTE: SSD variances are set in the anchors.py file
    anchorsC = Anchors()
    anchors = anchorsC.get_anchors(fmSizes=[(16, 16), (8, 8)], fmBased=True)

    data_loader = DataService(train_data, BATCH_SIZE, 32, workers=3)
    data_loader.start()
    epoch_Steps = len(train_data)//BATCH_SIZE
    step = 0
    # maxWidth = {}
    # ratio = {}
    storePath = './tmpAImg'
    if not os.path.exists(storePath):
        os.makedirs(storePath)
    for k in range(2):
        print('epoch:', k)

        flag = True
        while True:
            imgs, lbls = data_loader.pop()
            print('step:', step)
            for i in range(BATCH_SIZE):
                img = (imgs[i]*255).astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                boxes = np.array(lbls[i])*IM_S
                drawBoxes(img, boxes)
                locs, confs, pos_ind = anchorFillter_test(anchors, lbls[i], 0.3, 0.5)
                tmpIndexes = np.where(confs > 0)
                locIndexes = tmpIndexes[0].tolist()

                if not tesEnDe:
                    # 原始anchor
                    tmpLoc = (anchors*IM_S).astype(np.int32)
                elif len(locIndexes) > 0:
                    # 测试encode-decode
                    tmpAnchors = anchors[tmpIndexes]
                    print('tmpAnchors:', tmpAnchors.shape, tmpAnchors)
                    tmpLoc = decode_test(tmpAnchors, np.array(lbls[i])[pos_ind])*IM_S
                    print('tmpLoc:', tmpLoc.shape, tmpLoc)

                if len(locIndexes) > 0:
                    tmpBox = []
                    tmp = -1
                    if not tesEnDe:
                        for k in locIndexes:
                            if tmp != -1 and k - tmp < 10:
                                continue
                            tmp = k
                            tmpBox.append(tmpLoc[k])
                    else:
                        tmpBox = tmpLoc
                    drawBoxes(img, tmpBox, True)
                cv2.imwrite(os.path.join(storePath, str(uuid.uuid4())+'.jpg'), img)
            if step != 0 and step % epoch_Steps == 0:
                step += 1
                break

            step += 1
        break
    data_loader.stop()

    # print('dump data to Txt')
    # fmwh = open('fmwh.txt', 'w', encoding='utf-8')
    # for key in maxWidth.keys():
    #     fmwh.write(str(key)+'\t'+str(maxWidth[key])+'\n')
    # fmwh.close()
    # fratio = open('fratio.txt', 'w', encoding='utf-8')
    # for key in ratio.keys():
    #     fratio.write(str(key) + '\t' + str(ratio[key]) + '\n')
    # fratio.close()
    # data_loader.stop()
    # print('finish')
