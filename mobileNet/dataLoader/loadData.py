# !/usr/bin/python
# encoding:utf-8

'''
dataloader
'''

import tensorflow as tf

def getData(train_filenames, processFunc, param=None, batch_size=64, buffer_size=32):
    train_dataset = tf.data.Dataset.from_tensor_slices(train_filenames).map(
        lambda x: processFunc(x)).shuffle(buffer_size=buffer_size).batch(batch_size)

    iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    example_batch, label_batch = iterator.get_next()
    train_init_op = iterator.make_initializer(train_dataset)
    return example_batch, label_batch, train_init_op
