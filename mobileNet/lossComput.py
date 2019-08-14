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
    print('pos_AnchorInds:', np.squeeze(np.where(max_obj_iou_ids >= 0)))
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

    return locs, np.squeeze(targets)


def decode(anchor_boxes, locs, confs, min_conf=0.05, keep_top=400, nms_thresh=0.3, do_nms=True):
    # NOTE: confs is a N x 2 matrix
    # global SCALE_FACTOR

    centers_a = np.array(anchor_boxes[:, 2:] + anchor_boxes[:, :2]) / 2
    w_h_a = np.array(anchor_boxes[:, 2:] - anchor_boxes[:, :2])

    cxcy_in = locs[:, :2]/10.
    wh_in = locs[:, 2:]/5.

    wh = np.exp(wh_in) * w_h_a
    cxcy = cxcy_in * w_h_a + centers_a

    boxes_out = np.concatenate([cxcy - wh / 2, cxcy + wh / 2], axis=-1)

    # Get only if confidence > 0.05 & keep top 400 boxes
    # tmp = confs[:, 1]
    conf_ids = np.squeeze(np.argwhere(confs[:, 1] > min_conf))
    conf_merge = np.reshape(np.stack((conf_ids, confs[conf_ids, 1]), axis=-1), (-1, 2))
    conf_merge = conf_merge[conf_merge[:, 1].argsort()[::-1]]
    conf_merge = conf_merge[:keep_top, :]
    conf_ids, conf_vals = conf_merge[:, 0].astype(int), conf_merge[:, 1]
    # Run NMS on extracted boxes
    boxes_out = boxes_out[np.array(conf_merge[:, 0], dtype=int)]
    if do_nms:
        keep = non_max_suppression(boxes_out, nms_thresh)
        return boxes_out[keep], conf_ids[keep], conf_vals[keep]
    else:
        return boxes_out, conf_ids, conf_vals


def decode_batch(anchors, locs, confs, min_conf=0.5):
    out_boxes = []
    for i in range(len(locs)):
        b, _, _ = decode(anchors, np.squeeze(locs[i]), np.squeeze(confs[i]), min_conf=min_conf, do_nms=True)
        out_boxes.append(b)
    return out_boxes


def encode_batch(anchors, boxes, minThresh=0.3, maxThresh=0.7):
    out_locs = []
    out_confs = []
    for i in range(len(boxes)):
        l, c = anchorFillter(anchors, boxes[i], minThresh, maxThresh)
        out_locs.append(l)
        out_confs.append(c)
    return np.array(out_locs), np.array(out_confs)[:, :, np.newaxis]


def smooth_L1_loss(y_true, y_pred):
    absolute_loss = tf.abs(y_true - y_pred)
    square_loss = 0.5 * (y_true - y_pred) ** 2
    l1_loss = tf.where(tf.less(absolute_loss, 1.0), square_loss, absolute_loss - 0.5)
    return tf.reduce_sum(l1_loss, axis=-1)


def generateAttentionMap(batch_size, shapes, gBoxes):
    attentions1 = []
    attentions2 = []
    minLen = (32 + 32 // 2) / 256.

    for k in range(batch_size):
        tmpAttention1 = []
        tmpAttention2 = []
        L1 = 0
        L2 = 0
        for i in range(len(shapes)):
            fmShape = shapes[i]
            attention_gt = np.zeros(fmShape, dtype=np.float32)
            attW, attH = fmShape
            for box in gBoxes[k]:
                if i == 0 and (box[2] > minLen or box[3] > minLen):
                    continue
                elif i != 0 and box[2] <= minLen and box[3] <= minLen:
                    continue
                if i == 0:
                    L1 += 1
                else:
                    L2 += 1
                x1 = (box[0] - box[2] / 2)*attW
                y1 = (box[1] - box[3] / 2)*attH
                x2 = min((box[0] + box[2] / 2)*attW, attW)
                y2 = min((box[1] + box[3] / 2)*attH, attH)
                attention_gt[max(int(y1), 0):min(math.ceil(y2), attH),
                                max(int(x1), 0):min(math.ceil(x2), attW)] = 1
            if i == 0:
                tmpAttention1.append(attention_gt)
            else:
                tmpAttention2.append(attention_gt)
        print('attentionMap num:', L1, L2)
        attentions1.append(np.array(tmpAttention1))
        attentions2.append(np.array(tmpAttention2))
    return np.squeeze(np.array(attentions1)), np.squeeze(np.array(attentions2))


def focal_loss(targets, plogits):
    alpha_factor = tf.ones((tf.shape(targets)[0], tf.shape(targets)[1]), dtype=tf.float32) * alpha
    alpha_factor = tf.where(targets[:, :, 1] == 1., alpha_factor, 1. - alpha_factor)
    focal_weight = tf.where(targets[:, :, 1] == 1., 1. - tf.nn.softmax(plogits)[:, :, 1], tf.nn.softmax(plogits)[:, :, 1])
    focal_weight = alpha_factor * tf.pow(focal_weight, gamma)
    bce = tf.nn.softmax_cross_entropy_with_logits_v2(logits=plogits, labels=targets)
    cls_loss = focal_weight * bce
    return cls_loss


def hard_negative_mining(conf_loss, pos_ids, batch_size=32, mult=3, min_negs=10):
    with tf.name_scope('hard_negative_mining'):
        pos_ids = tf.unstack(pos_ids)
        neg_ids = [tf.logical_not(p) for p in pos_ids]
        conf_loss = tf.unstack(conf_loss)
        loss_out = []
        for c_i in range(batch_size):
            c_pos_ids = pos_ids[c_i]
            c_neg_ids = neg_ids[c_i]
            c_num_neg = tf.cast(tf.reduce_sum(tf.cast(c_neg_ids, tf.float32)), tf.int32)
            c_num_pos = tf.cast(tf.reduce_sum(tf.cast(c_pos_ids, tf.float32)), tf.int32)
            c_conf_loss = conf_loss[c_i]
            # c_l1_loss = l1_loss[c_i]
            loss_conf_neg = tf.reshape(tf.boolean_mask(c_conf_loss, c_neg_ids),
                                       [c_num_neg])  # Extract negative confidence losses only
            loss_conf_pos = tf.reshape(tf.boolean_mask(c_conf_loss, c_pos_ids), [c_num_pos])
            # loss_l1_pos = tf.reshape(tf.boolean_mask(c_l1_loss, c_pos_ids), [c_num_pos])
            c_neg_cap = tf.cast(mult * c_num_pos, tf.int32)
            c_neg_cap = tf.maximum(min_negs, c_neg_cap)  # Cap minimum negative value to min_negs
            c_neg_cap = tf.minimum(c_neg_cap, c_num_neg)  # Cap minimum values to max # = anchor_len
            loss_conf_k_neg, _ = tf.nn.top_k(loss_conf_neg, k=c_neg_cap, sorted=True)
            loss_out.append(tf.concat((loss_conf_pos, loss_conf_k_neg), axis=0))
        return tf.concat(loss_out, axis=0)


def faceDetLoss(plogits, pBoxes, locs_true, confs_true, batch_size=32, pAttention=None, attention_gt=None, match_threshold=0.5,
                negative_ratio=3., scope='faceDetLoss'):
    with tf.name_scope(scope):
        loc_preds = tf.reshape(pBoxes, (batch_size, -1, 4))
        conf_preds = tf.reshape(plogits, (batch_size, -1, 2))
        loc_true = tf.reshape(locs_true, (batch_size, -1, 4))
        conf_true = tf.cast(tf.reshape(confs_true, (batch_size, -1)), tf.int32)
        conf_true_oh = tf.one_hot(conf_true, 2)

        cls_check = tf.reshape(tf.cast(tf.greater_equal(conf_true, 0), tf.float32), (batch_size, tf.shape(loc_preds)[1]))
        # n_cls = tf.maximum(tf.reduce_sum(cls_check), 1)

        positive_check = tf.reshape(tf.cast(tf.equal(conf_true, 1), tf.float32), (batch_size, tf.shape(loc_preds)[1]))
        pos_ids = tf.cast(positive_check, tf.bool)
        n_pos = tf.maximum(tf.reduce_sum(positive_check), 1)

        l1_loss = tf.losses.huber_loss(loc_true, loc_preds, reduction=tf.losses.Reduction.NONE)  # Smoothed L1 loss
        l1_loss = positive_check * tf.reduce_sum(l1_loss, axis=-1)  # Zero out L1 loss for negative boxes

        cls_loss = focal_loss(conf_true_oh, conf_preds)
        cls_loss = cls_check*cls_loss
        # cls_loss = hard_negative_mining(cls_loss, pos_ids, batch_size)

        # loss = (cls_loss + tf.reduce_sum(l1_loss))/n_pos

        attLoss = 0.
        if pAttention is not None:
            for i in range(len(pAttention)):
                if not pAttention[i]:
                    continue
                tmpLoss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pAttention[i], labels=attention_gt[i]))
                if i == 0:
                    attLoss = tmpLoss
                else:
                    attLoss += tmpLoss
            attLoss = attLoss/len(pAttention)
            # loss += attLoss
        return tf.reduce_sum(l1_loss)/n_pos, tf.reduce_sum(cls_loss)/n_pos, attLoss if attLoss is not None else 0. # /float(batch_size)


if __name__ == '__main__':
    anchors = np.zeros(shape=(250, 4), dtype=np.float32)
    boxes = np.ones((25, 4), dtype=np.float32)
    gBoxes = np.ones((3, 4), dtype=np.float32)
    plogits = np.zeros(shape=[25, 2], dtype=np.float32)
    import random
    for i in range(250):
        for j in range(4):
            if i < 3:
                gBoxes[i][j] = random.random()
            if i < 25:
                boxes[i][j] = random.random()
                if j == 0:
                    plogits[i][0] = random.random()
                    plogits[i][1] = 1 - plogits[i][0]
            else:
                anchors[i][j] = random.random()

    # boxes = tf.constant(boxes.tolist())
    anchorFillter(anchors, gBoxes, 0., 0.01)
    # faceDetLoss(plogits, boxes, anchors, gBoxes, match_threshold=0., batch_size=1)
