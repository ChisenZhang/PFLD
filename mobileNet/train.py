# !/usr/bin/python
# encoding:utf-8

'''
训练代码
'''

import tensorflow as tf
import numpy as np
from mobileNet import MobileNetV2
from anchors import Anchors
import sys

import os
import time

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

import cv2
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "7"  # gpu编号
config = tf.ConfigProto()
config.gpu_options.allow_growth = True # 设置最小gpu使用量

sys.path.append('./dataLoader')
sys.path.append('./dataLoader/WIDE_FACE')

from WIDE_FACE.wider_voc import VOCDetection, AnnotationTransform
from WIDE_FACE.data_augment import preproc
from dataLoader import DataService

data_train_dir = '/home/wei.ma/face_detection/FaceBoxes.PyTorch/data/WIDER_FACE/'
data_test_dir = '/home/wei.ma/face_detection/FaceBoxes.PyTorch/data/FDDB'

# Training parameters
optimizer = 'adam'        # A string from: 'agd', 'adam', 'momentum'
learning_rate = 5e-3
momentum = None          # Necessary if optimizer is 'momentum'
summarize = True         # True if summarize in tensorboard
step = 0


def count_number_trainable_params(scope=""):
    '''
    Counts the number of trainable variables.
    '''
    tot_nb_params = 0
    vars_chk = None
    if scope == "":
        vars_chk = tf.trainable_variables()
    else:
        vars_chk = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    for trainable_variable in vars_chk:
        shape = trainable_variable.get_shape() # e.g [D,F] or [W,H,C]
        current_nb_params = get_nb_params_shape(shape)
        tot_nb_params = tot_nb_params + current_nb_params
    return tot_nb_params

def get_nb_params_shape(shape):
    '''
    Computes the total number of params for a given shape.
    Works for any number of shapes etc [D,F] or [W,H,C] computes D*F and W*H*C.
    '''

    nb_params = 1
    for dim in shape:
        nb_params = nb_params*int(dim)
    return nb_params

def dataProcess(img_id):
    # img_id = self.ids[index]
    print('process: ', img_id[1])
    _annopath = os.path.join(data_train_dir, 'annotations', '%s')
    _imgpath = os.path.join(data_train_dir, 'images', '%s')
    target = ET.parse(_annopath % img_id[1]).getroot()
    #print ("self._imgpath % image_id[0]", self._imgpath % img_id[0])
    #print ("target: ", target)
    img = cv2.imread(_imgpath % img_id[0], cv2.IMREAD_COLOR)
    height, width, _ = img.shape

    target = AnnotationTransform(target)

    img, target = preproc(img, target)

    return img, target

def getBatch(examples, batch_size):
    global globalExpIndex
    num = len(examples)
    if globalExpIndex + batch_size >= num:
        globalExpIndex = 0
        print('reset GEXPIndex:', globalExpIndex)
        random.shuffle(examples)
        return None, None
    tmpBatch = examples[globalExpIndex:globalExpIndex+batch_size]
    globalExpIndex += batch_size
    print('GEXPIndex:', globalExpIndex)
    return tmpBatch[0], tmpBatch[1]


if __name__ == '__main__':
    np.set_printoptions(suppress=True)

    timeCur = time.strftime('%y%m%d%H%M%S', time.localtime(time.time()))
    save_f = './models/'
    if not os.path.exists(save_f):
        os.mkdir(save_f)
    model_name = 'faceDet_fmBaseAnchor_'+timeCur+'_'
    PRINT_FREQ = 10
    TEST_FREQ = 1000
    SAVE_FREQ = 5000
    BATCH_SIZE = 32
    MAX_EPOCH = 300
    IM_S = 256
    IM_CHANNELS = 3
    IOU_THRESH = 0.5
    USE_NORM = True

    # NOTE: SSD variances are set in the anchors.py file
    anchorsC = Anchors()
    anchors = anchorsC.get_anchors(fmSizes=[(16, 16), (8, 8)], fmBased=True)

    tf.reset_default_graph()

    train_data = VOCDetection(data_train_dir, preproc(IM_S, 0, 1/255.), AnnotationTransform())
    print('train_wideFaceNum:', len(train_data))

    data_loader = DataService(train_data, BATCH_SIZE, 32, workers=5)
    data_loader.start()
    epoch_Steps = len(train_data)//BATCH_SIZE

    print('Building model...')
    # x = tf.placeholder(dtype=tf.float32, shape=[None, 256, 256, 3], name='input')
    fd_model = MobileNetV2(num_classes=2, batch_size=BATCH_SIZE, anchorsLen=anchors.shape[0])
    fd_model.blazeModel(learning_rate, int((len(train_data) / BATCH_SIZE) / 4))

    # Summarize in tensorboard
    # if summarize:
    #     tf.summary.scalar('loss', loss)
    #     tf.summary.scalar('accuracy', accuracy)
    #     tf.summary.scalar('learning_rate', lr)
    #     tf.summary.image("input_img", example_batch, max_outputs=6)
    #     summary_op = tf.summary.merge_all()

    print('Num params: ', count_number_trainable_params())

    # example_batch, label_batch, train_init_op = getData(train_data.ids, dataProcess, batch_size=BATCH_SIZE)
    print('begain sess')
    with tf.Session(config=config) as sess:
        print('Attempting to find a save file...')
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5, keep_checkpoint_every_n_hours=2)
        try:
            ckpt = tf.train.get_checkpoint_state(save_f)
            if ckpt is None:
                raise IOError('No valid save file found')
            print('#####################')
            print(ckpt.model_checkpoint_path)
            saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Succesfully loaded saved model')
        except IOError:
            print('Model not found - using default initialisation!')
            sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter('./logs', sess.graph)
        train_mAP_pred = []
        train_loss = []
        test_mAP_pred = []

        for k in range(MAX_EPOCH):
            print('epoch:', k)

            flag = True
            while True:
                imgs, lbls = data_loader.pop()

                if imgs is None:
                    break

                print('Iteration ', step, ' ', end='\r')

                try:
                    loss, summary = fd_model.getTrainLoss(sess, imgs, anchors, lbls)
                    train_loss.append(loss)
                    # train_mAP_pred.append(mAP)
                    writer.add_summary(summary, step)
                    if step % PRINT_FREQ == 0:
                        print("")
                        print('Iteration: ', step, end='')
                        print(' Mean train loss: ', np.mean(train_loss), end='')
                        # print(' Mean train mAP: ', np.mean(train_mAP_pred))
                        # train_mAP_pred = []
                        train_loss = []
                    # if i%TEST_FREQ == 0:
                    #     for j in range(25):
                    #         imgs, lbls = svc_test.random_sample(BATCH_SIZE)
                    #         pred_confs, pred_locs = fb_model.test_iter(imgs)
                    #         pred_boxes = anchors.decode_batch(boxes_vec, pred_locs, pred_confs)
                    #         test_mAP_pred.append(anchors.compute_mAP(imgs, lbls, pred_boxes, normalised = USE_NORM))
                    #     print('Mean test mAP: ', np.mean(test_mAP_pred))
                    #     test_mAP_pred = []
                    if step % SAVE_FREQ == 0:
                        print('Saving model...')
                        saver.save(sess, save_f + model_name+str(step), global_step=step)
                    if step != 0 and step % epoch_Steps == 0:
                        step += 1
                        break
                except Exception as E:
                    print('run error:', E)
                    data_loader.stop()
                    exit(-1)
                step += 1

            print('Saving epoch model...')
            saver.save(sess, save_f + model_name + str(step), global_step=step)
            print("epoch %d, steps %d" % (k, step))
        data_loader.stop()
        saver.save(sess, save_f + model_name + str(step)+'_final', global_step=step)
        print('trained finish!')
