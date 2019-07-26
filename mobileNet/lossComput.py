# !/usr/bin/python
# encoding:utf-8

'''
loss 计算函数
'''

import tensorflow as tf
import numpy as np
import math

alpha = 0.25
gamma = 2

def compute_iou_np(bboxes1, bboxes2):
    # Extracted from: https://medium.com/@venuktan/vectorized-intersection-over-union-iou-in-numpy-and-tensor-flow-4fa16231b63d
    x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))

    interArea = np.maximum(0, yB - yA) * np.maximum(0, xB - xA)

    boxAArea = (x12 - x11) * (y12 - y11)
    boxBArea = (x22 - x21) * (y22 - y21)

    # Fix divide by 0 errors
    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea + 0.00001)
    return np.clip(iou, 0, 1)

def non_max_suppression(boxes, overlapThresh):
    # Extracted from: https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1) * (y2 - y1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return pick

# function pick = nms(boxes,threshold,type)
def nms(boxes, threshold, method='Union'):
    if boxes.size == 0:
        return np.empty((0, 3))
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    s = boxes[:, 4] # 置信度
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    I = np.argsort(s)
    pick = np.zeros_like(s, dtype=np.int16)
    counter = 0
    while I.size > 0:
        i = I[-1]
        pick[counter] = i
        counter += 1
        idx = I[0:-1]
        xx1 = np.maximum(x1[i], x1[idx])
        yy1 = np.maximum(y1[i], y1[idx])
        xx2 = np.minimum(x2[i], x2[idx])
        yy2 = np.minimum(y2[i], y2[idx])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if method is 'Min':
            o = inter / np.minimum(area[i], area[idx])
        else:
            o = inter / (area[i] + area[idx] - inter + 0.00001)
        I = I[np.where(o <= threshold)]
    pick = pick[0:counter]
    return pick

# function [dy edy dx edx y ey x ex tmpw tmph] = pad(total_boxes,w,h)
def pad(total_boxes, w, h):
    """Compute the padding coordinates (pad the bounding boxes to square)"""
    tmpw = (total_boxes[:, 2] - total_boxes[:, 0] + 1).astype(np.int32)
    tmph = (total_boxes[:, 3] - total_boxes[:, 1] + 1).astype(np.int32)
    numbox = total_boxes.shape[0]

    dx = np.ones((numbox), dtype=np.int32)
    dy = np.ones((numbox), dtype=np.int32)
    edx = tmpw.copy().astype(np.int32)
    edy = tmph.copy().astype(np.int32)

    x = total_boxes[:, 0].copy().astype(np.int32)
    y = total_boxes[:, 1].copy().astype(np.int32)
    ex = total_boxes[:, 2].copy().astype(np.int32)
    ey = total_boxes[:, 3].copy().astype(np.int32)

    tmp = np.where(ex > w)
    edx.flat[tmp] = np.expand_dims(-ex[tmp] + w + tmpw[tmp], 1)
    ex[tmp] = w

    tmp = np.where(ey > h)
    edy.flat[tmp] = np.expand_dims(-ey[tmp] + h + tmph[tmp], 1)
    ey[tmp] = h

    tmp = np.where(x < 1)
    dx.flat[tmp] = np.expand_dims(2 - x[tmp], 1)
    x[tmp] = 1

    tmp = np.where(y < 1)
    dy.flat[tmp] = np.expand_dims(2 - y[tmp], 1)
    y[tmp] = 1

    return dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph

# function [boundingbox] = bbreg(boundingbox,reg)
def bbreg(boundingbox, reg):
    """Calibrate bounding boxes"""
    if reg.shape[1] == 1:
        reg = np.reshape(reg, (reg.shape[2], reg.shape[3]))

    w = boundingbox[:, 2] - boundingbox[:, 0] + 1
    h = boundingbox[:, 3] - boundingbox[:, 1] + 1
    b1 = boundingbox[:, 0] + reg[:, 0] * w
    b2 = boundingbox[:, 1] + reg[:, 1] * h
    b3 = boundingbox[:, 2] + reg[:, 2] * w
    b4 = boundingbox[:, 3] + reg[:, 3] * h
    boundingbox[:, 0:4] = np.transpose(np.vstack([b1, b2, b3, b4]))
    return boundingbox

# function [bboxA] = rerec(bboxA)
# box 转成正方形
def rerec(bboxA):
    """Convert bboxA to square."""
    h = bboxA[:, 3] - bboxA[:, 1]
    w = bboxA[:, 2] - bboxA[:, 0]
    l = np.maximum(w, h)
    bboxA[:, 0] = bboxA[:, 0] + w * 0.5 - l * 0.5
    bboxA[:, 1] = bboxA[:, 1] + h * 0.5 - l * 0.5
    bboxA[:, 2:4] = bboxA[:, 0:2] + np.transpose(np.tile(l, (2, 1)))
    return bboxA


def anchorFillter(anchors, gBoxes, minThresh=0.3, maxThresh=0.7):
    gBoxes = np.array(gBoxes)
    ious = compute_iou_np(anchors, gBoxes)
    # 每个anchor的最大IOU及index
    # max_iou, max_iou_ids = np.max(ious, axis=0), np.argmax(ious, axis=0)
    # 每个box的最大IOU及index
    max_obj_iou, max_obj_iou_ids = np.max(ious, axis=1), np.argmax(ious, axis=1)

    targets = np.ones((anchors.shape[0], 2), dtype=np.int32)*-1
    max_obj_iou = np.squeeze(max_obj_iou)
    max_obj_iou_ids = np.squeeze(max_obj_iou_ids)

    targets[max_obj_iou < minThresh] = 0
    pos_inds = np.where(max_obj_iou >= maxThresh)
    pos_inds = np.squeeze(pos_inds)
    assertBoxes = gBoxes[max_obj_iou_ids]
    targets[pos_inds] = 0
    targets[pos_inds, 1] = 1

    return pos_inds, targets, assertBoxes

def smooth_L1_loss(y_true, y_pred):
    absolute_loss = tf.abs(y_true - y_pred)
    square_loss = 0.5 * (y_true - y_pred) ** 2
    l1_loss = tf.where(tf.less(absolute_loss, 1.0), square_loss, absolute_loss - 0.5)
    return tf.reduce_sum(l1_loss, axis=-1)

def focal_loss(targets, plogits):
    alpha_factor = np.ones(targets.shape[0], dtype=np.float32) * alpha
    alpha_factor = tf.where(targets[:, 1] == 1., alpha_factor, 1. - alpha_factor)
    focal_weight = tf.where(targets[:, 1] == 1., 1. - tf.nn.softmax(plogits)[:, 1], tf.nn.softmax(plogits)[:, 1])
    focal_weight = alpha_factor * tf.pow(focal_weight, gamma)
    bce = tf.nn.softmax_cross_entropy_with_logits(logits=plogits, labels=targets)
    cls_loss = tf.reduce_sum(focal_weight * bce)
    return cls_loss

def faceDetLoss(plogits, pBoxes, anchors, gBoxes, batch_size=32, pAttention=None, match_threshold=0.5,
                negative_ratio=3., scope='faceDetLoss'):
    with tf.name_scope(scope):
        loss = 0.
        for k in range(batch_size):
            pos_inds, targets, assertBoxes = anchorFillter(anchors, gBoxes[k], maxThresh=match_threshold)
            cls_loss = focal_loss(targets, plogits[k])

            l1_loss = 0.
            if pos_inds.size > 0:
                assAnchors = anchors[pos_inds]
                # classes = tf.nn.softmax(plogits)
                # probs = classes[tuple(pos_inds), 1]
                pos_inds.dtype = 'int32'
                pBoxes = pBoxes[k, pos_inds]
                diff = (assertBoxes - assAnchors)
                diff[:, 0] /= assAnchors[:, 2]
                diff[:, 1] /= assAnchors[:, 3]
                diff[:, 2] /= assAnchors[:, 2]
                diff[:, 3] /= assAnchors[:, 3]
                l1_loss = tf.where(tf.abs(diff - pBoxes) < 1, 0.5*tf.pow(diff - pBoxes, 2),
                                   tf.abs(diff - pBoxes) - 0.5)
                l1_loss = tf.reduce_sum(l1_loss)

            attLoss = 0.
            if pAttention is not None:
                for i in range(len(pAttention)):
                    attention_gt = np.zeros(pAttention[i][k].shape, dtype=np.float32)
                    if i == 0:
                        attW = 16
                        attH = 16
                    elif i == 1:
                        attW = 8
                        attH = 8
                    else:
                        print('attention loss error:', pAttention.shape)
                        exit(-1)

                    # attW, attH, _ = pAttention[i][k].shape
                    for box in gBoxes[k]:
                        x1 = box[0] - box[2]/2
                        y1 = box[1] - box[3]/2
                        x2 = min(math.ceil(box[0] + box[2]/2) + 1, attention_gt.shape[0])
                        y2 = min(math.ceil(box[1] + box[3]/2) + 1, attention_gt.shape[1])
                        attention_gt[max(int(y1*attH), 0):min(math.ceil(y2*attH), attH),
                                        max(int(x1*attW), 0):min(math.ceil(x2*attW), attW)] = 1
                    attLoss += tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=pAttention[i][k],
                                                                                     labels=attention_gt, name='attLoss'))

            loss += cls_loss + l1_loss + attLoss

        return loss/float(batch_size)

if __name__ == '__main__':
    anchors = np.zeros(shape=(25, 4), dtype=np.float32)
    boxes = np.ones((1, 25, 4), dtype=np.float32)
    gBoxes = np.ones((1, 3, 4), dtype=np.float32)
    plogits = np.zeros(shape=[1, 25, 2], dtype=np.float32)
    import random
    for i in range(30):
        for j in range(4):
            if i < 3:
                gBoxes[0][i][j] = random.random()
            if i >= 25:
                pass
                # boxes[0][i][j] = random.random()
            else:
                boxes[0][i][j] = random.random()
                anchors[i][j] = random.random()
                if j == 0:
                    plogits[0][i][0] = random.random()
                    plogits[0][i][1] = 1 - plogits[0][i][0]
    boxes = tf.constant(boxes.tolist())
    # anchorFillter(anchors, boxes)
    faceDetLoss(plogits, boxes, anchors, gBoxes, match_threshold=0., batch_size=1)
