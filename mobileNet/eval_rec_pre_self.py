# !/usr/bin/python
# encoding:utf-8

'''
测试评价
'''

import numpy as np
import xml.etree.ElementTree as ET
import os
import json
import cv2

testAnchor = False

if testAnchor:
    from anchors import Anchors
    anchorsC = Anchors()
    anchors = anchorsC.get_anchors(fmSizes=[(16, 16), (8, 8)], fmBased=True, imgSize=1)

def parsingR(fileName):
    tmpDict = {}
    # tmp = []
    tmpTime = []
    with open(fileName, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace('\n', '')
            line = line.replace('.jpg', '')
            items = line.split('\t')
            imgName = items[0]
            tmpT = float(items[1]) # 0
            tmpTime.append(tmpT)
            tmpBox = []
            for i in range(2, len(items)): # 1
                if not items[i]:
                    break
                tmpXY = items[i].split(',')
                tmpBox.append((int(tmpXY[0]), int(tmpXY[1]), int(tmpXY[2]), int(tmpXY[3])))
            tmpDict[imgName] = [tmpT, tmpBox]
    print('evalT:', np.array(tmpTime).mean())
    return tmpDict


def parse_rec(gtPath):
    objects = []
    with open(gtPath, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            obj_struct = {}
            Jobj = json.loads(line[1:-1])
            x1 = Jobj["x"]
            y1 = Jobj["y"]
            x2 = x1+Jobj["w"]
            y2 = y1+Jobj["h"]
            obj_struct['bbox'] = [x1, y1, x2, y2]
            objects.append(obj_struct)
    return objects


def predicted_load(image_names, predicted):
    predicted_results = []
    for image_name in image_names:
        predicted_result_file = image_name
        for line in predicted[predicted_result_file][1]:
            x_min, x_max, y_min, y_max = line
            predicted_result = [image_name, 1., str(x_min), str(y_min), str(x_max), str(y_max)]
            predicted_results.append(predicted_result)
    return predicted_results


def compute_rec_pre(predicted,
                    gt_path,
                    image_set,
                    ovthresh=0.5,
                    dstSize=256.,
                    skipMinLenThresh=32):
    image_names = image_set

    # extract gt objects for this class
    npos = 0
    tmpL = predicted_load(image_set, predicted)
    # go down dets and mark TPs and FPs
    nd = len(image_names)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    fn = np.zeros(nd)
    stp = np.zeros(nd)
    sfp = np.zeros(nd)
    sfn = np.zeros(nd)
    d = 0
    skipNum = 0
    # f = open('FDDB_annotition.txt', 'w', encoding='utf-8')
    for image_name in image_names:
        # tmpS = image_name+'\t'
        objs = parse_rec(os.path.join(gt_path, image_name + '.json'))
        if not testAnchor:
            tmpImg = cv2.imread(os.path.join(gt_path, image_name+'.jpg'))
        else:
            tmpImg = cv2.imread(os.path.join('C:/Users/17ZY-HPYKFD2/Downloads/dFServer/tmpDetImgs_self_zheng0.5', image_name+'.jpg'))
        OSize = tmpImg.shape
        gt = [obj for obj in objs]
        bbox = np.array([x['bbox'] for x in gt])
        detctions = [False] * len(gt)
        npos = npos
        gt = {'bbox': bbox, 'det': detctions}

        pb = np.array(predicted[image_name][1]).astype(float)
        GT_box = gt['bbox'].astype(float)

        # resize后坐标转换
        # for i in range(len(GT_box)):
        #     GBox = GT_box[i]
        #     if dstSize:
        #         r = dstSize/max(OSize[0], OSize[1])
        #         GBox = (np.array(GBox) * r).astype(np.int32)
        #     # if resized is not None:
        #     #     bb = [int(bb[0] / float(resized[0]) * OSize[0]), int(bb[1] / float(resized[1]) * OSize[1]),
        #     #           int(bb[2] / float(resized[0]) * OSize[0]), int(bb[3] / float(resized[1]) * OSize[1])]
        #     # if skipMinLenThresh and (GBox[3] - GBox[1] < skipMinLenThresh*0.9 or GBox[2] - GBox[0] < skipMinLenThresh*0.9):
        #     #     continue
        #     tmpS += str(GBox[0]) + ',' + str(GBox[1]) + ',' + str(GBox[2]) + ',' + str(GBox[3]) + '\t'
        # tmpS = tmpS[:-1]+'\n'
        # f.write(tmpS)
        # continue

        small = 0
        for i in range(len(GT_box) if not testAnchor else len(anchors)):#):
            if testAnchor:
                GBox = anchors[i].astype(np.int32) # GT_box[i]
                dstSize = None
            else:
                GBox = GT_box[i]

            if dstSize:
                r = dstSize / max(OSize[0], OSize[1])
                GBox = (np.array(GBox) * r).astype(np.int32)
            # if resized is not None:
            #     bb = [int(bb[0] / float(resized[0]) * OSize[0]), int(bb[1] / float(resized[1]) * OSize[1]),
            #           int(bb[2] / float(resized[0]) * OSize[0]), int(bb[3] / float(resized[1]) * OSize[1])]

            tooSmall = False
            if skipMinLenThresh and (GBox[3] - GBox[1] < skipMinLenThresh or GBox[2] - GBox[0] < skipMinLenThresh):
                small += 1
                tooSmall = True

            # intersection
            ovmax = -np.inf
            jmax = -1
            for j in range(len(pb)):
                bb = pb[j]
                ixmin = np.maximum(GBox[0], bb[0])
                iymin = np.maximum(GBox[1], bb[1])
                ixmax = np.minimum(GBox[2], bb[2])
                iymax = np.minimum(GBox[3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih

                # union
                uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                       (GBox[2] - GBox[0] + 1.) *
                       (GBox[3] - GBox[1] + 1.) - inters)
                overlaps = inters / uni
                if overlaps > ovmax:
                    ovmax = overlaps
                    jmax = i

            if tooSmall:
                if ovmax > ovthresh:
                    if not gt['det'][jmax]:
                        stp[d] += 1
                        gt['det'][jmax] = 1
                    else:
                        sfp[d] += 1
                else:
                    sfn[d] += 1
            else:
                if ovmax > ovthresh:
                    if testAnchor:
                        tImg = tmpImg.copy()
                        cv2.rectangle(tImg, (GBox[0], GBox[1]), (GBox[2], GBox[3]), (0, 0, 255), 2)
                        cv2.imshow('tImg', tImg)
                        cv2.waitKey(0)
                    else:
                        if not gt['det'][jmax]:
                            tp[d] += 1
                            gt['det'][jmax] = 1
                        else:
                            fp[d] += 1
                else:
                    fn[d] += 1
        if tp[d] + fp[d] + stp[d] + sfp[d] < len(pb):
            fp[d] += len(pb) - (tp[d] + fp[d])
        d += 1

        if small > 0:
            skipNum += small
            npos -= small
    # f.close()
    print('skip Num:', skipNum)
    gt_num = npos
    predicted_num = tp.shape[0]
    true_predicted_num = np.sum(tp)
    false_predicted_num = np.sum(fp)
    false_negative_num = np.sum(fn)
    rec = true_predicted_num / (true_predicted_num + false_negative_num)
    prec = true_predicted_num / (true_predicted_num + false_predicted_num)
    print('small tp:%d, fp:%d, fn:%d' % (np.sum(stp), np.sum(sfp), np.sum(sfn)))
    print('small rec:%f, pre:%f' % (np.sum(stp)/(np.sum(stp) + np.sum(sfn)), np.sum(stp)/(np.sum(stp) + np.sum(sfp))))
    return rec, prec, true_predicted_num+false_negative_num, len(tmpL), true_predicted_num, false_predicted_num


if __name__ == '__main__':
    # import os, shutil
    # resultPath = 'C:/Users/17ZY-HPYKFD2/Downloads/dFServer/result_mtcnn_wider.txt'
    # preR, tmp = parsingR(resultPath)
    # xmlPath = 'D:/PyCode/wider-face-pascal-voc-annotations/WIDER_val_annotations'
    # for Head, file in tmp:
    #     if os.path.exists(os.path.join(xmlPath, file+'.xml')):
    #         newName = Head+'_'+file+'.xml'
    #         shutil.copyfile(os.path.join(xmlPath, file+'.xml'), os.path.join('C:/Users/17ZY-HPYKFD2/Downloads/dFServer/WIDER_val', newName))
    #     else:
    #         print('not found file:', Head, file)
    # exit(1)

    # gt_path = '../data/banbian_0815'
    # gt_path = '../data/cemian_0815'
    gt_path = '../data/zhengmian_0815'

    resultPath = 'C:/Users/17ZY-HPYKFD2/Downloads/dFServer/result_mobileNetSelf_'+gt_path.split('/')[-1 if gt_path[-1] != '/' else -2]+'.txt'
    # resultPath = 'C:/Users/17ZY-HPYKFD2/Downloads/dFServer/result_mtcnn_zheng.txt'
    # resultPath = 'C:/Users/17ZY-HPYKFD2/Downloads/dFServer/result_mobileNetWIDER_val.txt'

    # resultPath = 'C:/Users/17ZY-HPYKFD2/Downloads/dFServer/result_mtcnn_FDDB.txt'
    # resultPath = 'C:/Users/17ZY-HPYKFD2/Downloads/dFServer/result_mtcnn_wider.txt'

    preR = parsingR(resultPath)

    imgs = list(preR.keys())
    img_name_list = []
    for img in imgs:
        img_name = img # .strip().split('/')
        img_name_list.append(img_name)

    image_set = img_name_list
    ovthresh = 0.7

    recall, precision, gt_nums, predicted_nums, true_predicted_nums, \
    false_predicted_nums = compute_rec_pre(preR, gt_path, image_set, ovthresh,
                                           dstSize=256. if 'mtcnn' not in resultPath else None,
                                           skipMinLenThresh=32 if 'mtcnn' not in resultPath else None) # , resized=[256, 256])
    print('----------------------------------')
    # print('english word detection test:')
    print('gt_num: %d, detection_num: %d, tp: %d, fp: %d' % (int(gt_nums), int(predicted_nums), int(true_predicted_nums),
                                                                int(false_predicted_nums)))
    print('rec: %f, prec: %f, errDet:%f' % (recall, precision, false_predicted_nums / predicted_nums))
    print('----------------------------------')
