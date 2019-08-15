# !/usr/bin/python
# encoding:utf-8

'''
测试评价
'''

import numpy as np
import xml.etree.ElementTree as ET


def parsingR(fileName):
    tmpDict = {}
    with open(fileName, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace('\n', '')
            line = line.replace('.jpg', '')
            items = line.split('\t')
            imgName = items[0]
            imgName = imgName.split('/')[2:]
            imgName = '_'.join(imgName)
            tmpT = float(items[1]) # 0
            tmpBox = []
            for i in range(2, len(items)): # 1
                if not items[i]:
                    break
                tmpXY = items[i].split(',')
                tmpBox.append((int(tmpXY[0]), int(tmpXY[1]), int(tmpXY[2]), int(tmpXY[3])))
            tmpDict[imgName] = [tmpT, tmpBox]
    return tmpDict


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    size = None
    for tmp in tree.findall('size'):
        size = (int(tmp.find('width').text), int(tmp.find('height').text))

    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        # obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)

    return objects, size


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
    nd = len(tmpL)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    d = 0

    for image_name in image_names:
        objs, OSize = parse_rec(gt_path + image_name + '.xml')
        gt = [obj for obj in objs]
        bbox = np.array([x['bbox'] for x in gt])
        difficult = np.array([x['difficult'] for x in gt]).astype(np.bool)
        detctions = [False] * len(gt)
        npos = npos + sum(~difficult)
        gt = {'bbox': bbox, 'difficult': difficult, 'det': detctions}

        pb = np.array(predicted[image_name][1]).astype(float)
        GT_box = gt['bbox'].astype(float)
        for j in range(len(pb)):
            # intersection
            ovmax = -np.inf
            for i in range(len(GT_box)):
                bb = pb[j]
                GBox = GT_box[i]
                r = dstSize/max(OSize[0], OSize[1])
                # if resized is not None:
                #     bb = [int(bb[0] / float(resized[0]) * OSize[0]), int(bb[1] / float(resized[1]) * OSize[1]),
                #           int(bb[2] / float(resized[0]) * OSize[0]), int(bb[3] / float(resized[1]) * OSize[1])]
                GBox = (np.array(GBox)*r).astype(np.int32)
                if GBox[3] - GBox[1] < skipMinLenThresh or GBox[2] - GBox[0] < skipMinLenThresh:
                    continue
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

            if ovmax > ovthresh:
                if not gt['difficult'][jmax]:
                    if not gt['det'][jmax]:
                        tp[d] = 1.
                        gt['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.
            d += 1

    gt_num = npos
    predicted_num = tp.shape[0]
    true_predicted_num = np.sum(tp)
    false_predicted_num = np.sum(fp)
    rec = true_predicted_num / gt_num
    prec = true_predicted_num / predicted_num
    return rec, prec, gt_num, predicted_num, true_predicted_num, false_predicted_num


if __name__ == '__main__':
    gt_path = 'C:/Users/17ZY-HPYKFD2/Downloads/dFServer/FDDB/FDDB_xmlanno/'
    resultPath = 'C:/Users/17ZY-HPYKFD2/Downloads/dFServer/resultDet.txt'
    preR = parsingR(resultPath)

    imgs = list(preR.keys())
    img_name_list = []
    for img in imgs:
        img_name = img # .strip().split('/')
        img_name_list.append(img_name)

    image_set = img_name_list
    ovthresh = 0.5

    recall, precision, gt_nums, predicted_nums, true_predicted_nums, \
    false_predicted_nums = compute_rec_pre(preR, gt_path, image_set, ovthresh) # , resized=[256, 256])
    print('----------------------------------')
    print('english word detection test:')
    print('gt_num: %d, detection_num: %d, tp: %d, fp: %d'%(int(gt_nums), int(predicted_nums), int(true_predicted_nums),
                                                           int(false_predicted_nums)))
    print('rec: %f, prec: %f'%(recall, precision))
    print('----------------------------------')
