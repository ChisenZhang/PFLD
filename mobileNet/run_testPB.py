import sys
import cv2
import tensorflow as tf
import anchors
import numpy as np
import os
from lossComput import decode_batch


def lighting_balance(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

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


def main(argv):
    # pbModel_path = './models/pb/blazeFace_model_test.pb'
    pbModel_path = r'C:\Users\17ZY-HPYKFD2\Downloads\dFServer\blazeFace_model_test.pb'
    WIDTH_DES = 256
    HEIGHT_DES = 256
    USE_NORM = True
    UPSCALE = False
    anchorsC = anchors.Anchors()
    boxes_vec = anchorsC.get_anchors(fmSizes=[(16, 16), (8, 8)], fmBased=True)

    # Setup tensorflow and model
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Force on CPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force on CPU
    config = tf.ConfigProto()
    tf.reset_default_graph()
    with tf.Session(config=config) as sess:
        ret = True
        # Loop through video data
        while ret == True:
            # ret, frame = vid_in.read()
            frame = cv2.imread('./img_381.jpg')
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if UPSCALE:
                r = WIDTH_DES * 2 / frame.shape[1]
                dim_des = (int(WIDTH_DES * 2), int(frame.shape[0] * r))
                frame = cv2.resize(frame, dim_des, interpolation=cv2.INTER_LANCZOS4)
                c_shp = frame.shape
                frame = frame[int(c_shp[0] / 4):-int(c_shp[0] / 4),
                        int((c_shp[1] - WIDTH_DES) / 2):-int((c_shp[1] - WIDTH_DES) / 2)]
            else:
                r = WIDTH_DES / max(frame.shape[1], frame.shape[0])
                dim_des = (int(WIDTH_DES), int(frame.shape[1] * r))
                # frame = cv2.resize(frame, (WIDTH_DES, HEIGHT_DES))
                frame = cv2.resize(frame, (0, 0), fx=r, fy=r) # (WIDTH_DES, HEIGHT_DES))
                frame = np.pad(frame, ((0, HEIGHT_DES - frame.shape[0]), (0, WIDTH_DES - frame.shape[1]), (0, 0)))
            # frame_padded = lighting_balance(frame)
            # frame_padded = cv2.copyMakeBorder(frame, 0, max(0, HEIGHT_DES - frame.shape[0]), 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
            # pred_confs, pred_locs = model.test_iter(np.expand_dims(frame, axis = 0))
            tmp_frame = frame / 255.
            pred_locs, pred_confs = freeze_graph_test(pbModel_path, np.expand_dims(tmp_frame, axis=0))

            # f = open('paramR.txt', 'w', encoding='utf-8')
            # confT = pred_confs[0][:, 1]
            # for conf in confT:
            #     print(str(conf), file=f)
            # f.close()
            # exit(1)

            f = open('paramR.txt', 'w', encoding='utf-8')
            for i in range(896):
                l = pred_locs[0][i][0]
                t = pred_locs[0][i][1]
                r = pred_locs[0][i][2]
                b = pred_locs[0][i][3]
                p = pred_confs[0][i][1]
                print('index:', i, ', L:', l, ', T:', t, ', R:', r, ', B:', b, ', P:', p, file=f)
            f.close()

            pred_boxes = decode_batch(boxes_vec, pred_locs, pred_confs, min_conf=0.3)[0]
            pred_boxes[pred_boxes < 0] = 0
            pred_boxes[:, [0, 2]][pred_boxes[:, [0, 2]] > WIDTH_DES] = WIDTH_DES
            pred_boxes[:, [1, 3]][pred_boxes[:, [1, 3]] > HEIGHT_DES] = HEIGHT_DES
            h, w = HEIGHT_DES, WIDTH_DES
            for box in pred_boxes.tolist():
                if USE_NORM:
                    print(int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h))
                    cv2.rectangle(frame, (int(box[0] * w), int(box[1] * h)), (int(box[2] * w), int(box[3] * h)),
                                  (0, 255, 0), 3)
                    # cv2.rectangle(frame, (480, 72), (654, 294), (0, 255, 0), 3)
                else:
                    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 3)
            cv2.imshow('Webcam', frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite('./tmp.jpg', frame)
            cv2.waitKey(1)
            ret = False
        # vid_in.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main(sys.argv)

    # 测试缩放后坐标还原
    # frame = cv2.imread('./data/img_389.jpg')
    # w, h = frame.shape[1], frame.shape[0]
    # cv2.rectangle(frame, (int(480/1024.*w), int(72/1024.*h)), (int(654/1024.*w), int(294/1024.*h)), (0, 255, 0), 3)
    # cv2.imwrite('./data/dtmp.jpg', frame)