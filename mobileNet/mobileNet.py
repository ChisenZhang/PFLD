# !/usr/bin/python
# encoding:utf-8

'''
mobileNetV2网络结构
'''

import tensorflow as tf
# import numpy as np
# import time
from lossComput import faceDetLoss, encode_batch, generateAttentionMap
from anchors import Anchors


anchorsC = Anchors()
training_backBone = True
training_FaceDetection = True
training_LandMark = False

class MobileNetV2(object):
    def __init__(self, num_classes=2, batch_size=32, batchShape=(256, 256), training=True):
        self.input = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3], name='input')
        self.num_classes = num_classes
        if training:
            self.training = tf.placeholder(tf.bool, name="is_training")
        else:
            self.training = False
        self.batch_size = batch_size
        # self.anchorsLen = anchorsLen
        self.batch_shape = batchShape
        self.index = 0
        self.target_locs = tf.placeholder(tf.float32, shape=(None, None, 4), name='target_locs')
        self.target_confs = tf.placeholder(tf.float32, shape=(None, None, 1), name='target_confs')
        # self.target_ious = tf.placeholder(tf.float32, shape=(None, None, 1), name='target_ious')
        self.target_attention1 = tf.placeholder(tf.float32, shape=(None, 16, 16), name='target_attention1')
        self.target_attention2 = tf.placeholder(tf.float32, shape=(None, 8, 8), name='target_attention2')
        self.GStep = tf.Variable(0, trainable=False)
        self.attention1 = None
        self.attention2 = None

    def model(self, x):
        with tf.variable_scope('MobileNet'):
            output = tf.layers.conv2d(inputs=x,
                                      filters=16,
                                      kernel_size=5,
                                      strides=2,
                                      padding='same',
                                      name='conv1',
                                      trainable=training_backBone)
            output = tf.layers.batch_normalization(inputs=output, training=self.training)
            output = tf.nn.relu6(output, name='relu1')
            output = tf.layers.conv2d(inputs=output, filters=32, kernel_size=1, strides=1, padding='valid',
                                        name='conv2', trainable=training_backBone)
            output = tf.layers.batch_normalization(inputs=output, training=self.training)
            output = tf.nn.relu6(output, name='relu2')
            output = tf.layers.separable_conv2d(inputs=output, filters=32, kernel_size=3, strides=2, padding='same',
                                        name='conv3', trainable=training_backBone)
            output = tf.layers.batch_normalization(inputs=output, training=self.training)
            output = tf.nn.relu6(output, name='relu3')
            output = tf.layers.conv2d(inputs=output, filters=16, kernel_size=1, strides=1, padding='valid',
                                      activation=None, name='conv4', trainable=training_backBone)
            output = tf.layers.batch_normalization(inputs=output, training=self.training)

            output = tf.layers.conv2d(inputs=output, filters=32, kernel_size=3, strides=2, padding='same',
                                        name='conv5', trainable=training_backBone)
            output = tf.layers.batch_normalization(inputs=output, training=self.training)
            output = tf.nn.relu6(output, name='relu5')
            output = tf.layers.conv2d(inputs=output, filters=64, kernel_size=1, strides=1, padding='valid',
                                        name='conv6', trainable=training_backBone)
            output = tf.layers.batch_normalization(inputs=output, training=self.training)
            output = tf.nn.relu6(output, name='relu6')

            # self.output = self._inverted_bottleneck(output, 2, 16, 0)
            # self.output = self._inverted_bottleneck(self.output, 2, 16, 1)
            # self.output = self._inverted_bottleneck(self.output, 6, 24, 0)
            # self.output = self._inverted_bottleneck(self.output, 6, 32, 1)
            # self.output = self._inverted_bottleneck(self.output, 6, 32, 0)
            # self.output = self._inverted_bottleneck(self.output, 6, 32, 0)
            # self.output = self._inverted_bottleneck(self.output, 6, 64, 1)
            # self.output = self._inverted_bottleneck(self.output, 6, 64, 0)
            # self.output = self._inverted_bottleneck(self.output, 6, 64, 0)
            # self.output = self._inverted_bottleneck(self.output, 6, 64, 0)
            # self.output = self._inverted_bottleneck(self.output, 6, 96, 0)
            # self.output = self._inverted_bottleneck(self.output, 6, 96, 0)
            # self.output = self._inverted_bottleneck(self.output, 6, 96, 0)
            # self.output = self._inverted_bottleneck(self.output, 6, 160, 1)
            # self.output = self._inverted_bottleneck(self.output, 6, 160, 0)
            # self.output = self._inverted_bottleneck(self.output, 6, 160, 0)
            # self.output = self._inverted_bottleneck(self.output, 6, 320, 0)
            # self.output = tc.layers.conv2d(self.output, 64, 1, normalizer_fn=self.normalizer,
            #                                normalizer_params=self.bn_params)

            output = tf.layers.average_pooling2d(inputs=output, pool_size=16, strides=1, name='g_avePool')
            output = tf.layers.conv2d(inputs=output, filters=self.num_classes, kernel_size=1, strides=1,
                                      padding='valid', activation=None, name='conv7', trainable=training_backBone)
            self.logits = tf.reshape(output, shape=[-1, self.num_classes], name="logit")
            self.prob = tf.nn.softmax(self.logits, name='prob')

    def blazeModel(self, learning_rate, decay_step):
        with tf.variable_scope('BlazeNet'):
            # output = tf.image.resize_image_with_pad(self.input, 256, 256)
            # output = tf.image.resize_images(self.input, (256, 256), preserve_aspect_ratio=True,
            #                                          method=0)
            # output = tf.image.pad_to_bounding_box(output, 0, 0, 256, 256)
            output = self.input/255.
            output = tf.layers.conv2d(inputs=output,
                                      filters=16,
                                      kernel_size=5,
                                      strides=2,
                                      padding='same',
                                      name='conv1',
                                      trainable=training_backBone)
            output = tf.layers.batch_normalization(inputs=output, training=self.training)
            output = tf.nn.relu6(output, name='relu')
            output = self.BlazeBlock(output, 16, 1, 'BlazeBlock1', 1, False)
            output = self.BlazeBlock(output, 16, 1, 'BlazeBlock2', 1, False)
            output = self.BlazeBlock(output, 24, 1, 'BlazeBlock3', 2, False)
            output = self.BlazeBlock(output, 24, 1, 'BlazeBlock4', 1, False)
            output = self.BlazeBlock(output, 24, 1, 'BlazeBlock5', 1, False)
            output = self.BlazeBlock(output, 32, 1, 'BlazeBlock6', 2, True, 12)
            output = self.BlazeBlock(output, 32, 1, 'BlazeBlock7', 1, True, 16)
            output = self.BlazeBlock(output, 32, 1, 'BlazeBlock8', 1, True, 16)
            output1 = self.BlazeBlock(output, 48, 1, 'BlazeBlock9', 2, True, 16)
            output = self.BlazeBlock(output1, 48, 1, 'BlazeBlock10', 1, True, 24)
            output2 = self.BlazeBlock(output, 64, 1, 'BlazeBlock11', 2, True, 24)
            # out1 = self.inceptionBlock(output1)
            # out1, attention1 = self.attentionBlock(out1, scope='attentionBlock1')
            cls1, reg1 = self.clsAndReg(output1, self.num_classes, 8, scope='clsAndReg1')
            # out2 = self.inceptionBlock(output2)
            # out2, attention2 = self.attentionBlock(out2, scope='attentionBlock2')
            cls2, reg2 = self.clsAndReg(output2, self.num_classes, 24, scope='clsAndReg2')
            self.cls = tf.concat((tf.reshape(cls1, [self.batch_size, -1, self.num_classes]),
                                  tf.reshape(cls2, [self.batch_size, -1, self.num_classes])), axis=-2, name='cls')
            # self.cls = tf.reshape(self.cls, [self.batch_size, self.anchorsLen, self.num_classes], name='cls')
            self.prob = tf.nn.softmax(self.cls, name='probs')
            self.reg = tf.concat((tf.reshape(reg1, [self.batch_size, -1, 4]),
                                  tf.reshape(reg2, [self.batch_size, -1, 4])), axis=-2, name='reg')
            # self.reg = tf.reshape(self.reg, [self.batch_size, self.anchorsLen, 4], name='reg')
            # self.attention1 = tf.squeeze(attention1, name='attention1')
            # self.attention2 = tf.squeeze(attention2, name='attention2')

            # self.boundBoxes = tf.concat([self.target_locs[:, :, 0] - self.target_locs[:, :, 2]/2,
            #                           self.target_locs[:, :, 1] - self.target_locs[:, :, 3]/2,
            #                           self.target_locs[:, :, 0] + self.target_locs[:, :, 2] / 2,
            #                           self.target_locs[:, :, 1] + self.target_locs[:, :, 3] / 2], axis=-1)
            # self.boundBoxes = tf.reshape(self.boundBoxes, shape=[32, -1, 4])
            # self.drawImage = tf.image.draw_bounding_boxes(self.input, self.boundBoxes)
            # tf.summary.image("input_img", self.input, max_outputs=6)

            # 定义指数下降学习率
            self.lr = tf.train.exponential_decay(learning_rate=0.01, global_step=self.GStep,
                                                       decay_steps=10000, decay_rate=0.99, staircase=True)

            # 多项式衰减 往复
            # self.lr = tf.train.polynomial_decay(learning_rate=learning_rate, global_step=self.GStep,
            #                                decay_steps=decay_step, end_learning_rate=1e-8,
            #                                power=0.5, cycle=True)

            tf.summary.scalar('LR', self.lr)

            # L1_loss, cls_loss, attLoss = faceDetLoss(self.cls, self.reg, batch_size=self.batch_size, locs_true=self.target_locs,
            #                         confs_true=self.target_confs,
            #                         pAttention=[self.attention1, self.attention2],
            #                         attention_gt=[self.target_attention1, self.target_attention2])

            L1_loss, cls_loss, _ = faceDetLoss(self.cls, self.reg, batch_size=self.batch_size,
                                                     locs_true=self.target_locs,
                                                     confs_true=self.target_confs)
                                                     # ious_true=self.target_ious)

            self.loss = L1_loss + cls_loss # + attLoss if attLoss is not None else 0.
            self.loss += tf.losses.get_regularization_loss()  # Add regularisation
            tf.summary.scalar('L1_loss', L1_loss)
            tf.summary.scalar('cls_loss', cls_loss)
            # tf.summary.scalar('attLoss', attLoss)
            tf.summary.scalar('Loss', self.loss)
            self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(self.extra_update_ops):
                self.train = tf.train.AdamOptimizer(self.lr, epsilon=0.1).minimize(self.loss, global_step=self.GStep)
            self.merged = tf.summary.merge_all()

    def _inverted_bottleneck(self, input, up_sample_rate, channels, subsample, isTrainable=True):
        with tf.variable_scope('bottleneck_{}'.format(self.index)):
            self.index += 1
            stride = 2 if subsample else 1
            output = tf.layers.conv2d(input, up_sample_rate * input.get_shape().as_list()[-1], 1,
                                      activation=tf.nn.relu6, trainable=isTrainable)
            output = tf.layers.separable_conv2d(output, None, 3, 1, stride=stride,
                                                activation=tf.nn.relu6, trainable=isTrainable)
            output = tf.layers.batch_normalization(inputs=output, training=self.training)
            output = tf.layers.conv2d(output, channels, 1, activation_fn=None, trainable=isTrainable)
            if input.shape.as_list()[-1] == channels:
                output = tf.add(input, output)
            output = tf.layers.batch_normalization(inputs=output, training=self.training)
            return output

    # compExpRatio=0.25 通道压缩膨胀系数  默认中间通道压缩到原来的1/4。
    # <1 为中间层压缩通道, >1 为中间层通道膨胀
    def block(self, x, n_out, n_bottleneck, scope, strides=2, mode='deep', compExpRatio=0.25, isTrainable=True):
        with tf.variable_scope(scope):
            out = self.bottleneck(x, n_out=n_out, scope="bottleneck1", mode=mode, compExpRatio=compExpRatio,
                                  strides=strides, isTrainable=isTrainable)
            for i in range(1, n_bottleneck):
                out = self.bottleneck(out, n_out, scope="bottleneck{0}".format(i + 1), mode=mode,
                                      compExpRatio=compExpRatio, isTrainable=isTrainable)
        return out

    def bottleneck(self, x, n_out, scope, mode, strides=1, compExpRatio=0.25, isTrainable=True):
        n_in = x.shape[-1]
        n_first = int(n_out * compExpRatio)
        with tf.variable_scope(scope):
            if mode == 'shallow':
                f = tf.layers.separable_conv2d(inputs=x,
                                                 filters=n_out,
                                                 kernel_size=3,
                                                 strides=strides,
                                                 padding='same',
                                                 activation=tf.nn.relu,
                                                 name='conv1',
                                                 trainable=isTrainable
                                                 )
                f = tf.layers.batch_normalization(inputs=f, training=self.training)
                f = tf.layers.conv2d(inputs=f,
                                     filters=n_out,
                                     kernel_size=3,
                                     strides=1,
                                     padding='same',
                                     activation=None,
                                     name='conv2',
                                     trainable=isTrainable
                                     )
            elif mode == 'deep':
                f = tf.layers.conv2d(inputs=x,
                                     filters=n_first,
                                     kernel_size=1,
                                     strides=1,
                                     name='conv1',
                                     trainable=isTrainable
                                     )
                f = tf.layers.separable_conv2d(inputs=f,
                                                 filters=n_first,
                                                 kernel_size=3,
                                                 strides=strides,
                                                 padding='same',
                                                 name='conv2',
                                                 trainable=isTrainable
                                                 )
                f = tf.layers.batch_normalization(inputs=f, training=self.training)
                f = tf.layers.conv2d(inputs=f,
                                     filters=n_out,
                                     kernel_size=1,
                                     strides=1,
                                     name='conv3',
                                     trainable=isTrainable
                                     )
            else:
                raise ValueError("expected argument 'mode' must be between 'shallow' and 'deep'.")

            if n_in != n_out:  # projection
                shortcut = tf.layers.conv2d(inputs=x,
                                            filters=n_out,
                                            kernel_size=1,
                                            strides=strides,
                                            name='projection',
                                            trainable=isTrainable)
            else:
                shortcut = x  # identical mapping
            out = tf.layers.batch_normalization(inputs=shortcut+f, training=self.training)
            out = tf.nn.relu6(out)
            return out

    # blaze模块
    def BlazeBlock(self, x, n_out, n_bottleneck, scope, strides=1, doubleBLZ=False, compExpRatio=0.25, isTrainable=True):
        with tf.variable_scope(scope):
            out = self.Blaze(x, n_out=n_out, scope="Blaze1", doubleBLZ=doubleBLZ, compExpRatio=compExpRatio,
                                  strides=strides, isTrainable=isTrainable)
            for i in range(1, n_bottleneck):
                out = self.Blaze(out, n_out, scope="Blaze{0}".format(i + 1), doubleBLZ=doubleBLZ,
                                      compExpRatio=compExpRatio, isTrainable=isTrainable)
        return out

    def Blaze(self, x, n_out, scope, doubleBLZ=False, strides=1, compExpRatio=0.25, isTrainable=True):
        n_in = x.shape.as_list()[-1]
        n_first = compExpRatio # int(n_out * compExpRatio)
        with tf.variable_scope(scope):
            f = tf.layers.separable_conv2d(inputs=x,
                                             filters=n_in,
                                             kernel_size=5,
                                             strides=strides if not doubleBLZ else 1,
                                             padding='same',
                                             name='dconv1',
                                             trainable=isTrainable
                                             )
            f = tf.layers.batch_normalization(inputs=f, training=self.training)
            f = tf.layers.conv2d(inputs=f,
                                 filters=n_out if not doubleBLZ else n_first,
                                 kernel_size=1,
                                 strides=1,
                                 padding='same',
                                 name='pconv1',
                                 trainable=isTrainable
                                 )
            if doubleBLZ:
                f = tf.nn.relu6(f, name='relu')
                f = tf.layers.separable_conv2d(inputs=f,
                                               filters=n_first,
                                               kernel_size=5,
                                               strides=strides,
                                               padding='same',
                                               name='dconv2',
                                               trainable=isTrainable
                                               )
                f = tf.layers.batch_normalization(inputs=f, training=self.training)
                f = tf.layers.conv2d(inputs=f,
                                     filters=n_out,
                                     kernel_size=1,
                                     strides=1,
                                     padding='same',
                                     name='pconv2',
                                     trainable=isTrainable
                                     )
            # if strides != 1 or n_out - n_in > 0:
                # s = tf.layers.conv2d(inputs=x,
                #                     filters=n_out,
                #                     kernel_size=1,
                #                     strides=strides,
                #                     name='projection',
                #                     trainable=isTrainable)
            # else:
            #     s = x

            if strides != 1:
                s = tf.layers.max_pooling2d(x, pool_size=strides, strides=strides)
            else:
                s = x

            if n_out - n_in > 0:
                # tmpShape = s.shape.as_list()
                s = tf.pad(s, [[0, 0], [0, 0], [0, 0], [0, n_out - n_in]], name='padZero')
                # addShape = (self.batch_size, tmpShape[1], tmpShape[2], n_out - n_in)
                # addC = tf.zeros(shape=addShape, name='ZeroC', dtype=tf.float32)
                # s = tf.concat([s, addC], axis=-1, name='padZero')

            output = tf.layers.batch_normalization(inputs=f+s, training=self.training)
            output = tf.nn.relu6(output)
        return output

    # inception 模块
    def inceptionBlock(self, input, shortCut=True):
        group = input.shape[-1]//4
        g1 = input[:, :, :, 0:group]
        g2 = input[:, :, :, group:2*group]
        g3 = input[:, :, :, 2*group:3*group]
        g4 = input[:, :, :, 3*group:]
        print(g1.shape, g2.shape, g3.shape, g4.shape)
        g1out = tf.layers.conv2d(g1, filters=group, kernel_size=1, strides=1, padding='same', trainable=training_FaceDetection)
        g1out = tf.layers.conv2d(g1out, filters=group, kernel_size=[1, 3], strides=1, padding='same', trainable=training_FaceDetection)
        g1out = tf.layers.conv2d(g1out, filters=group, kernel_size=1, strides=1, padding='same', trainable=training_FaceDetection)
        g2out = tf.layers.conv2d(g2, filters=group, kernel_size=1, strides=1, padding='same', trainable=training_FaceDetection)
        g2out = tf.layers.conv2d(g2out, filters=group, kernel_size=[1, 5], strides=1, padding='same', trainable=training_FaceDetection)
        g2out = tf.layers.conv2d(g2out, filters=group, kernel_size=1, strides=1, padding='same', trainable=training_FaceDetection)
        g3out = tf.layers.conv2d(g3, filters=group, kernel_size=1, strides=1, padding='same', trainable=training_FaceDetection)
        g3out = tf.layers.conv2d(g3out, filters=group, kernel_size=[3, 1], strides=1, padding='same', trainable=training_FaceDetection)
        g3out = tf.layers.conv2d(g3out, filters=group, kernel_size=1, strides=1, padding='same', trainable=training_FaceDetection)
        g4out = tf.layers.conv2d(g4, filters=group, kernel_size=1, strides=1, padding='same', trainable=training_FaceDetection)
        g4out = tf.layers.conv2d(g4out, filters=group, kernel_size=[5, 1], strides=1, padding='same', trainable=training_FaceDetection)
        g4out = tf.layers.conv2d(g4out, filters=group, kernel_size=1, strides=1, padding='same', trainable=training_FaceDetection)
        out = tf.concat([g1out, g2out, g3out, g4out], axis=-1)
        out = tf.layers.conv2d(out, filters=input.shape[-1], kernel_size=1, strides=1, padding='same', trainable=training_FaceDetection)
        if shortCut:
            out = out+input
        out = tf.layers.batch_normalization(inputs=out, training=self.training)
        out = tf.nn.relu6(out)
        return out

    # 2个连续卷积
    def conv2(self, input, c_in, c_out, trainable=training_FaceDetection):
        output = tf.layers.conv2d(input, filters=c_in, kernel_size=3, strides=1, padding='same',
                                  trainable=trainable)
        output = tf.nn.relu6(output)
        output = tf.layers.conv2d(output, filters=c_out, kernel_size=3, strides=1, padding='same',
                                  trainable=trainable)
        output = tf.layers.batch_normalization(inputs=output, training=self.training)
        return output

    # 4个连续卷积
    def conv4(self, input, c_in, c_out, trainable=training_FaceDetection):
        output = tf.layers.conv2d(input, filters=c_in, kernel_size=3, strides=1, padding='same',
                                 trainable=trainable)
        output = tf.nn.relu6(output)
        output = tf.layers.conv2d(output, filters=c_in, kernel_size=3, strides=1, padding='same',
                                  trainable=trainable)
        output = tf.layers.batch_normalization(inputs=output, training=self.training)
        output = tf.nn.relu6(output)
        output = tf.layers.conv2d(output, filters=c_in, kernel_size=3, strides=1, padding='same',
                                  trainable=trainable)
        output = tf.nn.relu6(output)
        output = tf.layers.conv2d(output, filters=c_out, kernel_size=3, strides=1, padding='same',
                                  trainable=trainable)
        output = tf.layers.batch_normalization(inputs=output, training=self.training)
        return output

    # attention 模块
    def attentionBlock(self, input, scope='attentionBlock'):
        with tf.variable_scope(scope):
            channels = input.shape[-1]
            attention = self.conv4(input, channels, 1, training_FaceDetection)
            output = tf.exp(tf.nn.sigmoid(attention), name='attentionMap')
            output = tf.multiply(input, output)
            return output, attention

    # clssification and regression
    def clsAndReg(self, input, classes, anchors, addPosCls=0, addNegCls=0, scope='clsAndReg'):
        with tf.variable_scope(scope):
            channels = input.shape[-1]
            cls = self.conv2(input, channels, (classes + addPosCls + addNegCls)*anchors, trainable=training_FaceDetection)
            cls = tf.reshape(cls, shape=[self.batch_size, -1, (classes+addPosCls+addNegCls)], name='class')
            reg = self.conv2(input, channels, 4*anchors, trainable=training_FaceDetection)
            reg = tf.reshape(reg, shape=[self.batch_size, -1, 4], name='reg')
            return cls, reg

    # 返回loss
    def getTrainLoss(self, sess, imgs, batchShape, gBoxes):
        anchors = anchorsC.get_anchors(fmSizes=[(batchShape[1]//16, batchShape[0]//16),
                                                (batchShape[1]//32, batchShape[0]//32)], fmBased=True)
        print('anchors shape:', anchors.shape)
        locs, confs = encode_batch(anchors, gBoxes, 0.3, 0.5)
        # attention1, attention2 = generateAttentionMap(self.batch_size, shapes=[(16, 16), (8, 8)], gBoxes=gBoxes)
        # _, loss, merged, _ = sess.run([self.train, self.loss, self.merged, self.lr], feed_dict={self.input: imgs, self.target_locs: locs, self.target_confs: confs,
        #                                         self.target_attention1: attention1, self.target_attention2: attention2})
        _, loss, merged, _ = sess.run([self.train, self.loss, self.merged, self.lr],
                                      feed_dict={self.input: imgs, self.training: True, self.target_locs: locs, self.target_confs: confs}) # , self.target_ious: ious})
        return loss, merged

    def getResults(self, sess, imgs):
        probs, boxes = sess.run([self.prob, self.reg], feed_dict={self.input: imgs, self.training: False})

        return probs, boxes


if __name__ == '__main__':
    x = tf.placeholder('float', shape=[None, 256, 256, 3], name='x')
    net = MobileNetV2(2, batch_size=1, training=False)
    net.blazeModel(1e-3, 3)
    print('abc')
