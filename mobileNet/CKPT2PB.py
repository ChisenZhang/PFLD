# -*-coding: utf-8 -*-
"""
    @Project: tensorflow_models_nets
    @info:
    -通过传入 CKPT 模型的路径得到模型的图和变量数据
    -通过 import_meta_graph 导入模型中的图
    -通过 saver.restore 从模型中恢复图中各个变量的数据
    -通过 graph_util.convert_variables_to_constants 将模型持久化
"""

import tensorflow as tf
from tensorflow.python.framework import graph_util
import os
import time
import mobileNet
import utils
# from PIL import Image
import cv2
import numpy as np

resize_height = 256  # 指定图片高度
resize_width = 256  # 指定图片宽度
depths = 3

os.environ["CUDA_VISIBLE_DEVICES"] = "7"  # gpu编号
config = tf.ConfigProto()
config.gpu_options.allow_growth = True # 设置最小gpu使用量

def freeze_graph_test(pb_path, image_path):
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

            # 读取测试图片
            # im = cv2.imread(image_path)
            # im = cv2.cvtColor(cv2.resize(im, (resize_height, resize_width)), cv2.COLOR_BGR2RGB)/255.
            # im = im[np.newaxis, :]
            image = cv2.imread(image_path)
            im = cv2.resize(image, (256, 256))/255.
            # 测试读出来的模型是否正确，注意这里传入的是输出和输入节点的tensor的名字，不是操作节点的名字
            # out=sess.run("InceptionV3/Logits/SpatialSqueeze:0", feed_dict={'input:0': im,'keep_prob:0':1.0,'is_training:0':False})
            bt = time.time()
            probs, locs = sess.run([output_tensor_probs, output_tensor_locs], feed_dict={input_image_tensor: [im]})# ,
                                                          # input_keep_prob_tensor: 1.0,
                                                          # input_is_training_tensor: False})
            print("out:{}".format(probs), len(probs[0]), len(locs[0]), locs, time.time()-bt)
            # score = tf.nn.softmax(out, name='pre')
            class_id = tf.argmax(probs, 1)
            print("pre class_id:{}".format(sess.run(class_id)))

def run_test(pb_path, folder, classes=['ILLEGAL', 'NORMAL']):
    count = 0
    tp = 0.
    NORN = 0
    NOREN = 0.
    ILLN = 0
    ILLEN = 0.
    bt = time.time()
    tpH = time.time()
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
            input_image_tensor = sess.graph.get_tensor_by_name("x:0")
            # input_keep_prob_tensor = sess.graph.get_tensor_by_name("keep_prob:0")
            # input_is_training_tensor = sess.graph.get_tensor_by_name("is_training:0")

            # 定义输出的张量名称
            output_tensor_name = sess.graph.get_tensor_by_name("MobileNet/prob:0")
            for subFd in os.listdir(folder):
                subPath = os.path.join(folder, subFd)
                for fileN in os.listdir(subPath):
                    imgPath = os.path.join(subPath, fileN)
                    try:
                        image = cv2.imread(imgPath)
                        image_data = utils.process_image(image)
                    except Exception as E:
                        print('Open Error! ', imgPath, E)
                        continue
                    else:
                        picName = fileN.split('/')[-1]
                        cls = picName.split('_')[-1][:4]
                        if cls == 'NOR1':
                            cls = 'NORMAL'
                            NORN += 1
                        else:
                            cls = 'ILLEGAL'
                            ILLN += 1
                        t1 = time.time()
                        out = sess.run(output_tensor_name, feed_dict={input_image_tensor: image_data})
                        pred_class_index = np.argmax(out, axis=1)[0]
                        pred_prob = out[0, pred_class_index]
                        pred_class = classes[pred_class_index]
                        t2 = time.time()
                        if pred_class == cls:
                            tp += 1
                        else:
                            if cls == 'NORMAL':
                                NOREN += 1
                                # open(os.path.join(NOREP, picName), 'wb').write(open(os.path.join(tmPath, line), 'rb').read())
                            else:
                                ILLEN += 1
                                # open(os.path.join(ILLEP, picName), 'wb').write(open(os.path.join(tmPath, line), 'rb').read())
                            print(picName, cls, pred_class, pred_prob)
                        count += 1
                        if count % 100 == 0:
                            print(count, '\nTotal Recall:', round(float(tp / count) * 100, 2), 'Precession:',
                                  round(float(tp / (count + NOREN + ILLEN)) * 100, 2), '\nNormal Recall:',
                                  round((1 - float(NOREN / NORN)) * 100, 2) if NORN > 0 else None, 'Precession:',
                                  round((1 - float(NOREN / (NORN + ILLEN))) * 100, 2) if NORN + ILLEN > 0 else None,
                                  '\nIllegal Recall:',
                                  round((1 - float(ILLEN / ILLN)) * 100, 2) if ILLN > 0 else None, 'Precession:',
                                  round((1 - float(ILLEN / (ILLN + NOREN))) * 100, 2) if ILLN + NOREN > 0 else None,
                                  '\n' + str(round(time.time() - tpH, 2)))
                            tpH = time.time()
            et = time.time()
            print('finalR: ', count, '\nTotal Recall:', round(float(tp / count) * 100, 2), 'Precession:',
                  round(float(tp / (count + NOREN + ILLEN)) * 100, 2), '\nNormal Recall:',
                  round((1 - float(NOREN / NORN)) * 100, 2), 'Precession:',
                  round((1 - float(NOREN / (NORN + ILLEN))) * 100, 2), '\nIllegal Recall:',
                  round((1 - float(ILLEN / ILLN)) * 100, 2), 'Precession:',
                  round((1 - float(ILLEN / (ILLN + NOREN))) * 100, 2), '\n' + str(round(time.time() - tpH, 2)), et - bt)

def freeze_graph(input_checkpoint, output_graph):
    '''
    :param input_checkpoint:
    :param output_graph: PB模型保存路径
    :return:
    '''
    # checkpoint = tf.train.get_checkpoint_state(model_folder) #检查目录下ckpt文件状态是否可用
    # input_checkpoint = checkpoint.model_checkpoint_path #得ckpt文件路径

    # 指定输出的节点名称,该节点名称必须是原模型中存在的节点
    output_node_names = "BlazeNet/probs,BlazeNet/reg"
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)

    with tf.Session(config=config) as sess:
        saver.restore(sess, input_checkpoint)  # 恢复图并得到数据
        output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=sess.graph_def,  # 等于:sess.graph_def
            output_node_names=output_node_names.split(","))  # 如果有多个输出节点，以逗号隔开

        with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型
            f.write(output_graph_def.SerializeToString())  # 序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node))  # 得到当前图有几个操作节点

        for op in sess.graph.get_operations():
            print(op.name, op.values())

def freeze_graphOri(input_checkpoint, output_graph):
    # 指定输出的节点名称,该节点名称必须是原模型中存在的节点
    output_node_names = "BlazeNet/probs,BlazeNet/reg"
    net = mobileNet.MobileNetV2(num_classes=2, training=False, batch_size=1)
    net.blazeModel(learning_rate=1e-3, decay_step=50)
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        saver.restore(sess, tf.train.latest_checkpoint(input_checkpoint))  # 恢复图并得到数据
        output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=sess.graph_def,  # 等于:sess.graph_def
            output_node_names=output_node_names.split(","))  # 如果有多个输出节点，以逗号隔开

        with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型
            f.write(output_graph_def.SerializeToString())  # 序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node))  # 得到当前图有几个操作节点

def freeze_graph2(input_checkpoint, output_graph):
    '''
    :param input_checkpoint:
    :param output_graph: PB模型保存路径
    :return:
    '''
    # checkpoint = tf.train.get_checkpoint_state(model_folder) #检查目录下ckpt文件状态是否可用
    # input_checkpoint = checkpoint.model_checkpoint_path #得ckpt文件路径

    # 指定输出的节点名称,该节点名称必须是原模型中存在的节点
    output_node_names = "MobileNet/prob"
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    graph = tf.get_default_graph()  # 获得默认的图
    input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)  # 恢复图并得到数据
        output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=input_graph_def,  # 等于:sess.graph_def
            output_node_names=output_node_names.split(","))  # 如果有多个输出节点，以逗号隔开

        with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型
            f.write(output_graph_def.SerializeToString())  # 序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node))  # 得到当前图有几个操作节点

        # for op in graph.get_operations():
        #     print(op.name, op.values())

if __name__ == '__main__':
    # 输入ckpt模型路径
    input_checkpoint = './models'#/mobileNet_adam_190513175358.ckpt-3363.meta'
    # 输出pb模型的路径
    pb_path = "./models/pb"
    if not os.path.exists(pb_path):
        os.makedirs(pb_path)
    out_pb_path = os.path.join(pb_path, 'blazeFace_model_test.pb')

    # 调用freeze_graph将ckpt转为pb
    # freeze_graphOri(input_checkpoint, out_pb_path)

    # 测试pb模型
    image_path = './a.jpg'
    freeze_graph_test(pb_path=out_pb_path, image_path=image_path)

    # PB模型跑测试集
    # run_test(out_pb_path, '/data1/data/online_pushed_data/parse_result/illegalPicCls/testData')
