# !/usr/bin/python
# encoding:utf-8

'''
批量测试 自有数据集
'''

import cv2
import tensorflow as tf
import anchors
import numpy as np
import os
from lossComput import decode_batch
import time
import json
import sys

dstSize = 256
drawOriBox = True

def getGTBoxes(gtPath):
    GT_box = []
    with open(gtPath, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            Jobj = json.loads(line[1:-1])
            x1 = Jobj["x"]
            y1 = Jobj["y"]
            x2 = x1+Jobj["w"]
            y2 = y1+Jobj["h"]
            GT_box.append([x1, y1, x2, y2])
    return GT_box


def main(dataPath=None):
    pbModel_path = './models/pb/blazeFace_model_test.pb'
    # pbModel_path = r'C:\Users\17ZY-HPYKFD2\Downloads\dFServer\blazeFace_model_test.pb'
    if dataPath is not None:
        data_test_dir = dataPath
    else:
        data_test_dir = '/data1/image_data/data/faces/zhengmian_0815'
    # data_test_dir = '/data1/image_data/data/online_pushed_data/parse_result/illegalPicCls/NCNN/ncnn/WIDER_val'

    # lablePath = '/data1/image_data/data/online_pushed_data/parse_result/illegalPicCls/NCNN/ncnn/FDDB/FDDB_xmlanno'
    # lablePath = '/data1/image_data/data/online_pushed_data/parse_result/illegalPicCls/NCNN/ncnn/WIDER_val/xml'

    if not os.path.exists(data_test_dir):
        print('not found dataDir:', data_test_dir)
        exit(-1)

    # if 'FDDB' in data_test_dir:
    #     tail = 'FDDB'
    # else:
    tail = 'Self'

    storePath = './tmpDetImgs_self'

    if not os.path.exists(storePath):
        os.makedirs(storePath)
    else:
        os.system('rm -rf ' + storePath)
        os.makedirs(storePath)

    WIDTH_DES = 256
    HEIGHT_DES = 256
    anchorsC = anchors.Anchors()
    boxes_vec = anchorsC.get_anchors(fmSizes=[(16, 16), (8, 8)], fmBased=True)

    # Setup tensorflow and model
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Force on CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # gpu编号

    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(pbModel_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")
        with tf.Session() as sess:
            # 定义输入的张量名称,对应网络结构的输入张量
            # input:0作为输入图像,keep_prob:0作为dropout的参数,测试时值为1,is_training:0训练参数
            input_image_tensor = sess.graph.get_tensor_by_name("input:0")

            # 定义输出的张量名称
            output_tensor_probs = sess.graph.get_tensor_by_name("BlazeNet/probs:0")
            output_tensor_locs = sess.graph.get_tensor_by_name("BlazeNet/reg:0")

            f = open('result_mobileNetSelf_' + data_test_dir.split('/')[-1 if data_test_dir[-1] != '/' else -2] + '.txt', 'w', encoding='utf-8')

            for line in os.listdir(data_test_dir):
                if line.endswith('.jpg'):
                    print('process line:', line)
                    xmlPath = os.path.join(data_test_dir, line.split('.')[0] + '.json')
                    filePath = os.path.join(data_test_dir, line)
                    frame = cv2.imread(filePath)
                    OSize = frame.shape
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    r = WIDTH_DES / max(frame.shape[1], frame.shape[0])
                    # dim_des = (int(WIDTH_DES), int(frame.shape[1] * r))
                    # frame = cv2.resize(frame, (WIDTH_DES, HEIGHT_DES))
                    frame = cv2.resize(frame, (0, 0), fx=r, fy=r)  # (WIDTH_DES, HEIGHT_DES))
                    frame = np.pad(frame, ((0, HEIGHT_DES - frame.shape[0]), (0, WIDTH_DES - frame.shape[1]), (0, 0)),
                                   mode='constant')
                    tmp_frame = frame / 255.
                    bt = time.time()
                    pred_locs, pred_confs = sess.run([output_tensor_locs, output_tensor_probs],
                                                     feed_dict={input_image_tensor: np.expand_dims(tmp_frame, axis=0)})
                    totalT = time.time() - bt
                    pred_boxes = decode_batch(boxes_vec, pred_locs, pred_confs, min_conf=0.3)[0]
                    pred_boxes[pred_boxes < 0] = 0
                    # pred_boxes[:, [0, 2]][pred_boxes[:, [0, 2]] > WIDTH_DES] = WIDTH_DES
                    # pred_boxes[:, [1, 3]][pred_boxes[:, [1, 3]] > HEIGHT_DES] = HEIGHT_DES
                    h, w = HEIGHT_DES, WIDTH_DES
                    tmpS = line + '\t' + str(totalT) + '\t'

                    if drawOriBox:
                        GT_box = getGTBoxes(xmlPath)
                        for i in range(len(GT_box)):
                            GBox = GT_box[i]
                            if dstSize:
                                r = dstSize / max(OSize[0], OSize[1])
                                GBox = (np.array(GBox) * r).astype(np.int32)
                            cv2.rectangle(frame, (GBox[0], GBox[1]), (GBox[2], GBox[3]), (0, 0, 0), 3)

                    for box in pred_boxes.tolist():
                        tmpS += str(int(box[0] * w)) + ',' + str(int(box[1] * h)) + ',' + str(
                            int(box[2] * w)) + ',' + str(int(box[3] * h)) + '\t'
                        cv2.rectangle(frame, (int(box[0] * w), int(box[1] * h)), (int(box[2] * w), int(box[3] * h)),
                                      (0, 255, 0), 2)
                    tmpS = tmpS[:-1] + '\n'
                    f.write(tmpS)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    line = line.replace('/', '_')
                    cv2.imwrite(os.path.join(storePath, line), frame)
            f.close()
    os.system('zip -r tmpDetImgs.zip ' + storePath)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        dataPath = sys.argv[1]
    else:
        dataPath = None
    main(dataPath)