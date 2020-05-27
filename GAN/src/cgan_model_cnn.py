# -*- coding: utf-8 -*-
# @Author   : ZhengHj
# @Time     : 2018/12/23 18:18
# @Project  : testProject
# @File     : cgan_model_cnn.py
# @Software : PyCharm

import matplotlib.pyplot as plt
import numpy as np
import os
from six.moves import xrange
import pickle
import tensorflow as tf
from datetime import datetime

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('../datasets/mnist_datasets', one_hot=True)

noise_size = 100
smooth = 0.05
learning_rate = 0.001

# train
batch_size = 100
k = 10
epochs = 120


def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    y = tf.reshape(y, [-1, 1, 1, y_shapes[1]])
    y1 = y
    for i in range(x_shapes[1] - 1):
        y1 = tf.concat([y1, y], 1)
    y2 = y1
    print(y1.shape)
    for i in range(x_shapes[2] - 1):
        y2 = tf.concat([y2, y1], 2)
    print(y2.shape)
    return tf.concat([x, y2], 3)


def fully_connected(name, value, output_shape):
    with tf.variable_scope(name, reuse=None) as scope:
        shape = value.get_shape().as_list()
        w = tf.get_variable('w', [shape[1], output_shape], dtype=tf.float32,
                            initializer=tf.random_normal_initializer(stddev=0.01))
        b = tf.get_variable('b', [output_shape], dtype=tf.float32, initializer=tf.constant_initializer(0.0))

        return tf.matmul(value, w) + b



def lrelu(x, alpha=0.01):
    return tf.maximum(x, tf.multiply(x, alpha))

def get_noise(batch_size):
    return np.random.normal(0, 1, size=(batch_size, noise_size))

# def generator(noise, digits, keep_prob, reuse=False):
#     with tf.variable_scope('generator', reuse=reuse):
#         x = tf.concat([noise, digits], 1)
#         # (-1, 24)
#         x = tf.layers.dense(noise, units=49, activation=tf.nn.relu)
#         # (01, 49)
#         # x = tf.layers.dense(x, units=24 * 2 + 1, activation=tf.nn.relu)
#         x = tf.reshape(x, [-1, 7, 7, 1])
#         # (-1, 14, 14, 64)
#         x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=2, padding='same',
#                                        activation=tf.nn.relu)
#         # (-1, 14, 14, 64)
#         x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same',
#                                        activation=tf.nn.relu)
#         x = tf.nn.dropout(x, keep_prob)
#         # (-1, 14, 14, 64)
#         # x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same',
#         #                                activation=tf.nn.relu)
#         # (-1, 12544)
#         x = tf.contrib.layers.flatten(x)
#         # x = tf.concat([x, digits], 1)
#         # x = tf.layers.dense(x, 128, activation=tf.nn.relu)
#         x = tf.layers.dense(x, 784, activation=tf.nn.relu)
#         x = tf.layers.dense(x, units=28 * 28, activation=tf.nn.tanh)
#         img = tf.reshape(x, shape=[-1, 28, 28], name='img')
#     return img


def generator(noise_img, digit, keep_prob, reuse=False):
    with tf.variable_scope("generator", reuse=reuse):
        concatenated_img_digit = tf.concat([digit, noise_img], 1)

        #         output = tf.layers.dense(concatenated_img_digit, 256)

        output = fully_connected('gf1', concatenated_img_digit, 128)
        output = lrelu(output)
        output = tf.layers.dropout(output, rate=0.5)

        #         output = tf.layers.dense(output, 128)

        output = fully_connected('gf2', output, 128)
        output = lrelu(output)
        output = tf.layers.dropout(output, rate=0.5)

        #         logits = tf.layers.dense(output, 784)
        logits = fully_connected('gf3', output, 784)
        outputs = tf.tanh(logits)
        outputs = tf.reshape(outputs, (-1, 28, 28), name='img')

        return outputs


def discriminator(img, digits, keep_prob, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        img = tf.reshape(img, (-1, 28, 28, 1))
        img = conv_cond_concat(img, digits)
        # 第一层卷积层
        x = tf.layers.conv2d(img, filters=64, kernel_size=5, strides=1, padding='SAME', activation=lrelu)
        # 第二层卷积层
        x = tf.layers.conv2d(x, filters=64, kernel_size=5, strides=1, padding='SAME', activation=lrelu)
        # 最大值池化
        x = tf.layers.max_pooling2d(x, pool_size=2, strides=2, padding="SAME")
        # 二维图片变成一维
        x = tf.contrib.layers.flatten(x)
        # 第一层全连接层
        x = tf.layers.dense(x, 512, activation=lrelu)
        # dropout层
        x = tf.nn.dropout(x, keep_prob)
        # 第二层全连接层
        x = tf.layers.dense(x, 10, activation=lrelu)
        x = tf.concat([x, digits], 1)
        # 输出层
        logits = tf.layers.dense(x, 1, name='logits')
        outputs = tf.sigmoid(logits, name='outputs')

    return logits, outputs


# def discriminator(img, digit, keep_prob, reuse=False):
#     img = tf.reshape(img, (-1, 784))
#     with tf.variable_scope("discriminator", reuse=reuse):
#         concatenated_img_digit = tf.concat([digit, img], 1)
#
#         #         output = tf.layers.dense(concatenated_img_digit, 256)
#         output = fully_connected('df1', concatenated_img_digit, 128)
#         output = lrelu(output)
#         output = tf.layers.dropout(output, rate=0.5)
#
#         #         output = tf.layers.dense(concatenated_img_digit, 128)
#         output = fully_connected('df2', output, 128)
#         output = lrelu(output)
#         output = tf.layers.dropout(output, rate=0.5)
#
#         #         logits = tf.layers.dense(output, 1)
#         logits = fully_connected('df3', output, 1)
#         outputs = tf.sigmoid(logits, name='outputs')
#
#         return logits, outputs


def train_model():
    tf.reset_default_graph()
    # set placeholder
    real_img = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='real_img')
    noise_img = tf.placeholder(dtype=tf.float32, shape=[None, noise_size], name='noise_img')
    digits = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='digits')
    keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob')

    # build nets
    # generate fake_img
    fake_img = generator(noise_img, digits, keep_prob)

    # discriminate the true and false of the picture
    d_logits_real, d_outputs_real = discriminator(real_img, digits, keep_prob)
    d_logits_fake, d_outputs_fake = discriminator(fake_img, digits, keep_prob, reuse=True)

    # discriminator loss
    # d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,
    #                                                                      labels=tf.ones_like(d_logits_real)) * (1 - smooth))
    # d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
    #                                                                      labels=tf.zeros_like(d_logits_fake)))
    real_loss_d = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,
                                                          labels=tf.ones_like(d_logits_real)) * (1 - smooth)
    fake_loss_d = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                          labels=tf.zeros_like(d_logits_fake))
    d_loss_real = tf.reduce_mean((real_loss_d - 1) * (real_loss_d - 1))
    d_loss_fake = tf.reduce_mean(fake_loss_d * fake_loss_d)
    # loss
    d_loss = d_loss_real + d_loss_fake
    # generator loss
    # g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
    #                                                                 labels=tf.ones_like(d_logits_fake)) * (1 - smooth))
    gl1 = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                  labels=tf.ones_like(d_logits_fake)) * (1 - smooth)
    g_loss = tf.reduce_mean((gl1 - 1) * (gl1 - 1))

    # optimizer
    train_vars = tf.trainable_variables()

    # generator tensor
    g_vars = [var for var in train_vars if var.name.startswith("generator")]
    # discriminator tensor
    d_vars = [var for var in train_vars if var.name.startswith("discriminator")]

    # optimizer
    # d_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(d_loss, var_list=d_vars)
    d_train_opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(d_loss, var_list=d_vars)
    g_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(g_loss, var_list=g_vars)

    # configure gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9  # 程序最多只能占用指定gpu70%的显存
    config.gpu_options.allow_growth = True  # 程序按需申请内存

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        # Run the Op to initialize the variables.
        for e in xrange(epochs):
            print("{} Epoch {}/{} start".format(datetime.now(), e + 1, epochs))
            for i in xrange(mnist.train.num_examples // (batch_size * k)):
                for j in xrange(k):
                    batch = mnist.train.next_batch(batch_size)

                    real_img_digits = batch[1]

                    # scale the input images
                    images = batch[0]
                    images = 2 * images - 1  # 生成器激活函数tanh,(-1~1),把原始图像(0~1)也变为(-1~1)

                    # generator input noises
                    noises = get_noise(batch_size)

                    # Run optimizer
                    # sess.run([d_train_opt, g_train_opt],
                    #          feed_dict={real_img: images, noise_img: noises, digits: real_img_digits, keep_prob: 0.7})
                    _, dlr, dlf = sess.run([d_train_opt, d_loss_real, d_loss_fake],
                                           feed_dict={real_img: images, noise_img: noises, digits: real_img_digits, keep_prob: 0.7})
                    # print('d_loss_real: {}, d_loss_fake: {}'.format(dlr, dlf))
                    for _ in range(1):
                        _, gl = sess.run([g_train_opt, g_loss], feed_dict={noise_img: noises, digits: real_img_digits, keep_prob: 1.0})
                    # print('g_loss: {}'.format(gl))


            train_loss_d_real, train_loss_d_fake, train_loss_g = \
                sess.run([d_loss_real, d_loss_fake, g_loss],
                         feed_dict={real_img: images, noise_img: noises, digits: real_img_digits, keep_prob: 1.0})

            train_loss_d = train_loss_d_real + train_loss_d_fake

            print("Epoch {}/{}".format(e + 1, epochs),
                  "Discriminator loss: {:.4f}(Real: {:.4f} + Fake: {:.4f})".format(
                      train_loss_d, train_loss_d_real, train_loss_d_fake),
                  "Generator loss: {:.4f}".format(train_loss_g))
            size = 10
            sample_noise = get_noise(size)
            label = np.eye(10)
            labels = np.array(label)
            labels = labels.reshape(-1, 10)
            gen_samples = sess.run(fake_img, feed_dict={noise_img: sample_noise, digits: labels, keep_prob: 1.0})
            plt.figure(figsize=(19.2, 9.43))
            plt.rcParams['image.cmap'] = 'gray'
            for i in range(size):
                plt.subplot(2, 5, 1 + i)
                plt.imshow((gen_samples[i] + 1) / 2)
                plt.title(str(i))
                # plt.imshow(gen_samples[i].reshape(28, 28), cmap='Greys_r')
                # plt.show()
            title = "Real: {:.4f},  Fake: {:.4f}, Generator loss: {:.4f}".format(train_loss_d_real, train_loss_d_fake,
                                                                                  train_loss_g)
            plt.suptitle(title)
            # plt.suptitle('CGAN')
            plt.savefig('cgan_cnn_train_pictures/cgan_cnn_{}.png'.format(e+1), dpi=100, format='png')
            plt.close()

        saver = tf.train.Saver()
        # 保存模型
        saver.save(sess, 'cgan_cnn_model_save\\cgan_cnn.ckpt')


# 加载保存的模型
def usemodel(is_save=False, name='cgan_cnn'):
    tf.reset_default_graph()
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state('cgan_cnn_model_save')
        saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
        saver.restore(sess, ckpt.model_checkpoint_path)
        g = tf.get_default_graph()

        noise_img = g.get_tensor_by_name('noise_img:0')
        digits = g.get_tensor_by_name('digits:0')
        keep_prob = g.get_tensor_by_name('keep_prob:0')
        g_outputs = g.get_tensor_by_name('generator/img:0')

        size=10
        sample_noise = get_noise(size)
        labels=np.eye(10)
        gen_samples = sess.run(g_outputs, feed_dict={noise_img: sample_noise, digits: labels, keep_prob: 1.0})
        plt.figure(figsize=(19.2, 9.43))
        plt.rcParams['image.cmap'] = 'gray'
        for i in range(size):
            plt.subplot(2, 5, 1+i)

            # plt.imshow(gen_samples[i].reshape(28, 28))
            fake_img = (gen_samples[i] + 1) / 2
            plt.imshow(fake_img.reshape(28, 28))

            plt.title(str(i))
            # plt.imshow(gen_samples[i].reshape(28, 28), cmap='Greys_r')
            # plt.show()
        plt.suptitle('CGAN')
        if is_save:
            plt.savefig('cgan_cnn_pictures/' + name, dpi=100, format='png')
        else:
            plt.show()
        plt.close()


def use_model(is_save=False, times=1):
    for i in range(times):
        usemodel(is_save, 'cgan_cnn_{}.png'.format(i+1))


if __name__ == '__main__':
    # train_model()
    use_model(is_save=True, times=10)