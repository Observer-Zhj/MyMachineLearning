# -*- coding: utf-8 -*-
# @Author   : ZhengHj
# @Time     : 2018/12/22 21:22
# @Project  : testProject
# @File     : cvae_model.py
# @Software : PyCharm

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("../datasets/mnist_datasets", one_hot=True)

# tf.reset_default_graph()

batch_size = 64
dec_in_channels = 1
n_latent = 8
reshaped_dim = [-1, 7, 7, dec_in_channels]
inputs_decoder = int(49 * dec_in_channels / 2)


def lrelu(x, alpha=0.3):
    return tf.maximum(x, tf.multiply(x, alpha))


def encoder(X_in, digit, keep_prob):
    activation = lrelu
    with tf.variable_scope("encoder", reuse=None):
        X = tf.reshape(X_in, shape=[-1, 28, 28, 1])
        x = tf.layers.conv2d(X, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=1, padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, 784, activation=activation)
        # 卷积层之后合并标签digit，然后与全连接层连接
        x = tf.concat([x, digit], 1)
        x = tf.layers.dense(x, 64, activation=activation)
        mn = tf.layers.dense(x, units=n_latent)
        sd = 0.5 * tf.layers.dense(x, units=n_latent)
        epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], n_latent]))
        z = mn + tf.multiply(epsilon, tf.exp(sd))
        z = tf.add(z, 0, name='z')
    return z, mn, sd


def decoder(sampled_z, digit, keep_prob):
    with tf.variable_scope("decoder", reuse=None):
        x = tf.layers.dense(sampled_z, units=inputs_decoder, activation=lrelu)
        x = tf.layers.dense(x, units=inputs_decoder * 2 + 1, activation=lrelu)
        x = tf.reshape(x, reshaped_dim)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=2, padding='same',
                                          activation=tf.nn.relu)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same',
                                          activation=tf.nn.relu)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same',
                                          activation=tf.nn.relu)
        x = tf.contrib.layers.flatten(x)
        # 卷积层之后合并标签digit，然后与全连接层连接
        x = tf.concat([x, digit], 1)
        x = tf.layers.dense(x, 128, activation=lrelu)
        x = tf.layers.dense(x, 784, activation=lrelu)
        x = tf.layers.dense(x, units=28 * 28, activation=tf.nn.sigmoid)
        img = tf.reshape(x, shape=[-1, 28, 28], name='img')
    return img


def train_model():
    tf.reset_default_graph()
    X_in = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28], name='X')
    Y = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28], name='Y')
    Y_flat = tf.reshape(Y, shape=[-1, 28 * 28])
    digit = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='digit')
    keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob')
    sampled, mn, sd = encoder(X_in, digit, keep_prob)
    dec = decoder(sampled, digit, keep_prob)

    unreshaped = tf.reshape(dec, [-1, 28 * 28])
    img_loss = tf.reduce_sum(tf.squared_difference(unreshaped, Y_flat), 1)
    latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * sd - tf.square(mn) - tf.exp(2.0 * sd), 1)
    loss = tf.reduce_mean(img_loss + latent_loss)
    optimizer = tf.train.AdamOptimizer(0.0005).minimize(loss)

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9  # 程序最多只能占用指定gpu70%的显存
    config.gpu_options.allow_growth = True  # 程序按需申请内存
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    train_vars = tf.trainable_variables()

    # encoder tensor
    e_vars = [var for var in train_vars if var.name.startswith("encoder")]
    # decoder tensor
    d_vars = [var for var in train_vars if var.name.startswith("decoder")]

    for i in range(30000):
        batch = mnist.train.next_batch(batch_size=batch_size)
        images = [np.reshape(b, [28, 28]) for b in batch[0]]
        labels = batch[1]
        sess.run(optimizer, feed_dict={X_in: images, Y: images, digit: labels, keep_prob: 0.8})
        if not i % 200:
            ls, d, i_ls, d_ls, mu, sigm = sess.run([loss, dec, img_loss, latent_loss, mn, sd],
                                                   feed_dict={X_in: images, Y: images, digit: labels, keep_prob: 1.0})
            # plt.imshow(np.reshape(batch[0], [28, 28]), cmap='gray')
            # plt.show()
            # plt.imshow(d[0], cmap='gray')
            # plt.show()
            print(i, ls, np.mean(i_ls), np.mean(d_ls))

    saver = tf.train.Saver()
    saver.save(sess, 'cvae_model_save\\cvae.ckpt', global_step=i)
    sess.close()




# 加载保存的模型
def usemodel(is_save=False, name='cave'):
    tf.reset_default_graph()
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state('cvae_model_save')
        saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
        saver.restore(sess, ckpt.model_checkpoint_path)
        g = tf.get_default_graph()
        dec = g.get_tensor_by_name('decoder/img:0')
        sampled = g.get_tensor_by_name('encoder/z:0')
        keep_prob = g.get_tensor_by_name('keep_prob:0')
        digit = g.get_tensor_by_name('digit:0')

        randoms = [np.random.normal(0, 1, n_latent) for _ in range(10)]
        label = np.eye(10)
        labels = np.array(label)
        labels = labels.reshape(-1, 10)
        imgs = sess.run(dec, feed_dict={sampled: randoms, digit: labels, keep_prob: 1.0})
        imgs = [np.reshape(imgs[i], [28, 28]) for i in range(len(imgs))]
        plt.figure(figsize=(19.2, 9.43))
        plt.rcParams['image.cmap'] = 'gray'
        for i in range(len(imgs)):
            plt.subplot(2, 5, 1+i)
            plt.imshow(imgs[i].reshape(28, 28))
            plt.title(str(i))
            # plt.imshow(gen_samples[i].reshape(28, 28), cmap='Greys_r')
            # plt.show()
        plt.suptitle('CVAE')
        if is_save:
            plt.savefig('cvae_pictures/' + name, dpi=100, format='png')
        else:
            plt.show()
        plt.close()


def use_model(is_save=False, times=1):
    for i in range(times):
        usemodel(is_save, 'cvae_{}.png'.format(i+1))


if __name__ == '__main__':
    # train_model()
    use_model(is_save=True, times=10)
