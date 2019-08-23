# !/usr/bin/python
# encoding:utf-8

'''
批量测试
'''

import cv2
import tensorflow as tf
import anchors
import numpy as np
import os
from lossComput import decode_batch
import time
from eval_rec_pre import parse_rec
import sys
sys.path.append('./dataLoader')
sys.path.append('./dataLoader/WIDE_FACE')

from WIDE_FACE.wider_voc import VOCDetection, AnnotationTransform
from WIDE_FACE.data_augment import preproc
from dataLoader import DataService

dstSize = 256
drawOriBox = True

def freeze_graph_test(pb_path, im):
    '''
    :param pb_path:pb文件的路径
    :param image_path:测试图片的路径
    :return:
    '''
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(pb_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")
        with tf.Session() as sess:
            # sess.run(tf.global_variables_initializer())

            # 定义输入的张量名称,对应网络结构的输入张量
            # input:0作为输入图像,keep_prob:0作为dropout的参数,测试时值为1,is_training:0训练参数
            input_image_tensor = sess.graph.get_tensor_by_name("input:0")
            # input_keep_prob_tensor = sess.graph.get_tensor_by_name("keep_prob:0")
            # input_is_training_tensor = sess.graph.get_tensor_by_name("is_training:0")

            # 定义输出的张量名称
            output_tensor_probs = sess.graph.get_tensor_by_name("BlazeNet/probs:0")
            output_tensor_locs = sess.graph.get_tensor_by_name("BlazeNet/reg:0")

            # 输出相关层的结果
            # output_nodes = sess.graph.get_tensor_by_name("Conv1/Conv2D:0")
            # output = sess.run([output_nodes], feed_dict={input_image_tensor: im})
            # f = open('../class_tf/tmplog2.txt', 'w', encoding='utf-8')
            # for s in range(2):
            #     if s == 0:
            #         tmpO = im[0]
            #     else:
            #         tmpO = output[0][0]
            #     print('shape:', im.shape if s == 0 else output[0].shape, file=f)
            #     for i in range(tmpO.shape[0]):
            #         for j in range(tmpO.shape[1]):
            #             for k in range(tmpO.shape[2]):
            #                 print(str(tmpO[i][j][k])+', ', end='', file=f)
            #             print('\n', end='', file=f)
            #         print('\n\n', end='', file=f)
            # f.close()

            boxes, probs = sess.run([output_tensor_locs, output_tensor_probs], feed_dict={input_image_tensor: im})# ,
                                                          # input_keep_prob_tensor: 1.0,
                                                          # input_is_training_tensor: False})
            return boxes, probs


def getGTBoxes(gtPath):
    objs, OSize = parse_rec(gtPath)
    gt = [obj for obj in objs]
    bbox = np.array([x['bbox'] for x in gt])
    difficult = np.array([x['difficult'] for x in gt]).astype(np.bool)
    detctions = [False] * len(gt)
    gt = {'bbox': bbox, 'difficult': difficult, 'det': detctions}
    GT_box = gt['bbox'].astype(float)
    return GT_box, OSize


data_train_dir = '/home/wei.ma/face_detection/FaceBoxes.PyTorch/data/WIDER_FACE/'
train_data = VOCDetection(data_train_dir, preproc(dstSize, 0, 1/255.), AnnotationTransform())


def main():
    pbModel_path = './models/pb/blazeFace_model_test.pb'
    # pbModel_path = r'C:\Users\17ZY-HPYKFD2\Downloads\dFServer\blazeFace_model_test.pb'
    # data_test_dir = '/data1/image_data/data/online_pushed_data/parse_result/illegalPicCls/NCNN/ncnn/FDDB'
    # data_test_dir = '/data1/image_data/data/online_pushed_data/parse_result/illegalPicCls/NCNN/ncnn/WIDER_val'

    # lablePath = '/data1/image_data/data/online_pushed_data/parse_result/illegalPicCls/NCNN/ncnn/FDDB/FDDB_xmlanno'
    # lablePath = '/data1/image_data/data/online_pushed_data/parse_result/illegalPicCls/NCNN/ncnn/WIDER_val/xml'

    # if not os.path.exists(data_test_dir):
    #     print('not found dataDir:', data_test_dir)
    #     exit(-1)
    #
    # if 'FDDB' in data_test_dir:
    #     tail = 'FDDB'
    # else:
    #     tail = 'WIDER_val'

    # tail = 'train_data'
    storePath = './tmpDetImgs'
    if not os.path.exists(storePath):
        os.makedirs(storePath)
    else:
        os.system('rm -rf '+storePath)
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

            # with open(os.path.join(data_test_dir, 'img_list.txt'), 'r', encoding='utf-8') as f:
            #     lines = f.readlines()
            data_loader = DataService(train_data, 1, 32, workers=1)
            data_loader.start()
            # f = open('result_mobileNet'+tail+'.txt', 'w', encoding='utf-8')
            line = 0
            while True:
                if line > 50:
                    break
                imgs, lbls = data_loader.pop()
                # print('process line:', line)
                bt = time.time()
                pred_locs, pred_confs = sess.run([output_tensor_locs, output_tensor_probs], feed_dict={input_image_tensor: imgs})
                totalT = time.time() - bt
                pred_boxes = decode_batch(boxes_vec, pred_locs, pred_confs, min_conf=0.5)[0]
                pred_boxes[pred_boxes < 0] = 0
                # pred_boxes[:, [0, 2]][pred_boxes[:, [0, 2]] > WIDTH_DES] = WIDTH_DES
                # pred_boxes[:, [1, 3]][pred_boxes[:, [1, 3]] > HEIGHT_DES] = HEIGHT_DES
                h, w = HEIGHT_DES, WIDTH_DES
                # tmpS = '../FDDB/'+line+'.jpg\t'+str(totalT)+'\t'
                frame = (imgs[0]*255).astype(np.uint8)
                if drawOriBox:
                    for i in range(len(lbls[0])):
                        GBox = lbls[0][i]
                        # if dstSize:
                        #     r = dstSize / max(OSize[0], OSize[1])
                        GBox = (np.array(GBox) * dstSize).astype(np.int32)
                        cv2.rectangle(frame, (GBox[0], GBox[1]), (GBox[2], GBox[3]), (0, 0, 0), 3)

                for box in pred_boxes.tolist():
                    # tmpS += str(int(box[0] * w))+','+str(int(box[1] * h))+','+str(int(box[2] * w))+','+str(int(box[3] * h))+'\t'
                    cv2.rectangle(frame, (int(box[0] * w), int(box[1] * h)), (int(box[2] * w), int(box[3] * h)),
                                  (0, 255, 0), 2)
                # tmpS = tmpS[:-1] + '\n'
                # f.write(tmpS)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                # line = line.replace('/', '_')
                cv2.imwrite(os.path.join(storePath, str(line)+'.jpg'), frame)
                line += 1
            data_loader.stop()
            # f.close()
    os.system('zip -r tmpDetImgs.zip '+storePath)


if __name__ == "__main__":
    main()