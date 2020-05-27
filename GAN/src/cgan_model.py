# -*- coding: utf-8 -*-
# @Author   : ZhengHj
# @Time     : 2018/12/19 21:42
# @Project  : testProject
# @File     : cgan_model.py
# @Software : PyCharm

import matplotlib.pyplot as plt
import numpy as np
import os
from six.moves import xrange
import pickle
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('../datasets/mnist_datasets', one_hot=True)

if not os.path.exists("logdir"):
    os.makedirs("logdir")

LOGDIR = "logdir"
real_img_size = mnist.train.images[0].shape[0]
noise_size = 100
noise = 'normal0-1'
alpha = 0.1
learning_rate = 0.001
smooth = 0.05
# train
batch_size = 100
k = 10
epochs = 120


def leakyRelu(x, alpha=0.01):
    return tf.maximum(x, alpha * x)


def get_inputs(real_img_size, noise_size):
    real_img = tf.placeholder(tf.float32,
                              shape=[None, real_img_size], name="real_img")

    real_img_digit = tf.placeholder(tf.float32, shape=[None, k])

    noise_img = tf.placeholder(tf.float32,
                               shape=[None, noise_size], name="noise_img")

    return real_img, noise_img, real_img_digit


def fully_connected(name, value, output_shape):
    with tf.variable_scope(name, reuse=None) as scope:
        shape = value.get_shape().as_list()
        w = tf.get_variable('w', [shape[1], output_shape], dtype=tf.float32,
                            initializer=tf.random_normal_initializer(stddev=0.01))
        b = tf.get_variable('b', [output_shape], dtype=tf.float32, initializer=tf.constant_initializer(0.0))

        return tf.matmul(value, w) + b


def get_noise(noise, batch_size):
    if noise == 'uniform':
        batch_noise = np.random.uniform(-1, 1, size=(batch_size, noise_size))
    elif noise == 'normal':
        batch_noise = np.random.normal(-1, 1, size=(batch_size, noise_size))
    elif noise == 'normal0-1':
        batch_noise = np.random.normal(0, 1, size=(batch_size, noise_size))
    elif noise == 'uniform0-1':
        batch_noise = np.random.normal(0, 1, size=(batch_size, noise_size))

    return batch_noise


def get_generator(digit, noise_img, reuse=False):
    with tf.variable_scope("generator", reuse=reuse):
        concatenated_img_digit = tf.concat([digit, noise_img], 1)

        #         output = tf.layers.dense(concatenated_img_digit, 256)

        output = fully_connected('gf1', concatenated_img_digit, 128)
        output = leakyRelu(output)
        output = tf.layers.dropout(output, rate=0.5)

        #         output = tf.layers.dense(output, 128)

        output = fully_connected('gf2', output, 128)
        output = leakyRelu(output)
        output = tf.layers.dropout(output, rate=0.5)

        #         logits = tf.layers.dense(output, 784)
        logits = fully_connected('gf3', output, 784)
        outputs = tf.tanh(logits)

        return logits, outputs


def get_discriminator(digit, img, reuse=False):
    with tf.variable_scope("discriminator", reuse=reuse):
        concatenated_img_digit = tf.concat([digit, img], 1)

        #         output = tf.layers.dense(concatenated_img_digit, 256)
        output = fully_connected('df1', concatenated_img_digit, 128)
        output = leakyRelu(output)
        output = tf.layers.dropout(output, rate=0.5)

        #         output = tf.layers.dense(concatenated_img_digit, 128)
        output = fully_connected('df2', output, 128)
        output = leakyRelu(output)
        output = tf.layers.dropout(output, rate=0.5)

        #         logits = tf.layers.dense(output, 1)
        logits = fully_connected('df3', output, 1)
        outputs = tf.sigmoid(logits)

        return logits, outputs


def save_genImages(gen, epoch):
    r, c = 10, 10
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen[cnt][:, :], cmap='Greys_r')
            axs[i, j].axis('off')
            cnt += 1
    if not os.path.exists('gen_mnist1'):
        os.makedirs('gen_mnist1')
    fig.savefig('gen_mnist1/%d.jpg' % epoch)
    plt.close()


def plot_loss(loss):
    fig, ax = plt.subplots(figsize=(20, 7))
    losses = np.array(loss)
    plt.plot(losses.T[0], label="Discriminator Loss")
    plt.plot(losses.T[1], label="Discriminator_real_loss")
    plt.plot(losses.T[2], label="Discriminator_fake_loss")
    plt.plot(losses.T[3], label="Generator Loss")
    plt.title("Training Losses")
    plt.legend()
    plt.savefig('loss1.jpg')
    plt.show()


def Save_lossValue(e, epochs, train_loss_d, train_loss_d_real, train_loss_d_fake, train_loss_g):
    with open('loss1.txt', 'a') as f:
        f.write("Epoch {}/{}".format(e + 1, epochs),
                "Discriminator loss: {:.4f}(Real: {:.4f} + Fake: {:.4f})".format(
                    train_loss_d, train_loss_d_real, train_loss_d_fake),
                "Generator loss: {:.4f}".format(train_loss_g))


tf.reset_default_graph()

real_img, noise_img, real_img_digit = get_inputs(real_img_size, noise_size)

# generator
g_logits, g_outputs = get_generator(real_img_digit, noise_img)

sample_images = tf.reshape(g_outputs, [-1, 28, 28, 1])
tf.summary.image("sample_images", sample_images, 10)

# discriminator
d_logits_real, d_outputs_real = get_discriminator(real_img_digit, real_img)
d_logits_fake, d_outputs_fake = get_discriminator(real_img_digit, g_outputs, reuse=True)

# discriminator loss
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,
                                                                     labels=tf.ones_like(d_logits_real)) * (1 - smooth))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                                     labels=tf.zeros_like(d_logits_fake)))
# loss
d_loss = tf.add(d_loss_real, d_loss_fake)
# generator loss
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                                labels=tf.ones_like(d_logits_fake)) * (1 - smooth))

tf.summary.scalar("d_loss_real", d_loss_real)
tf.summary.scalar("d_loss_fake", d_loss_fake)
tf.summary.scalar("d_loss", d_loss)
tf.summary.scalar("g_loss", g_loss)

# optimizer
train_vars = tf.trainable_variables()

# generator tensor
g_vars = [var for var in train_vars if var.name.startswith("generator")]
# discriminator tensor
d_vars = [var for var in train_vars if var.name.startswith("discriminator")]

# optimizer
d_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(d_loss, var_list=d_vars)
g_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(g_loss, var_list=g_vars)

summary = tf.summary.merge_all()

saver = tf.train.Saver()


def train():
    # 保存loss值
    losses = []
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9  # 程序最多只能占用指定gpu70%的显存
    config.gpu_options.allow_growth = True  # 程序按需申请内存
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter(LOGDIR, sess.graph)

        # Run the Op to initialize the variables.
        for e in xrange(epochs):
            for i in xrange(mnist.train.num_examples // (batch_size * k)):
                for j in xrange(k):
                    batch = mnist.train.next_batch(batch_size)

                    digits = batch[1]

                    # scale the input images
                    images = batch[0].reshape((batch_size, 784))
                    images = 2 * images - 1  # 生成器激活函数tanh,(-1~1),把原始图像(0~1)也变为(-1~1)

                    # generator input noises
                    noises = get_noise(noise, batch_size)

                    # Run optimizer
                    sess.run([d_train_opt, g_train_opt], feed_dict={real_img: images, noise_img: noises, real_img_digit: digits})

            # train loss
            #         images = 2 * mnist.train.images - 1.0
            #         noises = get_sample([mnist.train.num_examples, noise_size])
            #         digits = mnist.train.labels

            summary_str, train_loss_d_real, train_loss_d_fake, train_loss_g = \
                sess.run([summary, d_loss_real, d_loss_fake, g_loss],
                         feed_dict={real_img: images, noise_img: noises, real_img_digit: digits})

            train_loss_d = train_loss_d_real + train_loss_d_fake
            losses.append((train_loss_d, train_loss_d_real, train_loss_d_fake, train_loss_g))

            summary_writer.add_summary(summary_str, e)
            summary_writer.flush()

            print("Epoch {}/{}".format(e + 1, epochs),
                  "Discriminator loss: {:.4f}(Real: {:.4f} + Fake: {:.4f})".format(
                      train_loss_d, train_loss_d_real, train_loss_d_fake),
                  "Generator loss: {:.4f}".format(train_loss_g))

            # # 保存模型
            # saver.save(sess, 'checkpoints\\cgan.ckpt', global_step=e+1)

            # 查看每轮结果

            size = 10
            sample_noise = get_noise(noise, size)
            # label = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0] * size
            label = np.eye(10)
            labels = np.array(label)
            labels = labels.reshape(-1, 10)
            _, gen_samples = sess.run(get_generator(real_img_digit, noise_img, reuse=True),
                                      feed_dict={noise_img: sample_noise, real_img_digit: labels})
            plt.figure(figsize=(19.2, 9.43))
            plt.rcParams['image.cmap'] = 'gray'
            for i in range(size):
                plt.subplot(2, 5, 1 + i)

                # plt.imshow(gen_samples[i].reshape(28, 28))
                fake_img = (gen_samples[i] + 1) / 2
                plt.imshow(fake_img.reshape(28, 28))

                plt.title(str(i))
                # plt.imshow(gen_samples[i].reshape(28, 28), cmap='Greys_r')
                # plt.show()
            title = "Real: {:.4f},  Fake: {:.4f}, Generator loss: {:.4f}".format(train_loss_d_real, train_loss_d_fake, train_loss_g)
            plt.suptitle(title)
            plt.savefig('cgan_train_pictures/cgan_{}.png'.format(e+1), dpi=100, format='png')
            plt.close()

            # gen_sample = get_noise(noise, batch_size)
            # label = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0] * 100  # 给定标签条件生成制定的数字
            # labels = np.array(label)
            # labels = labels.reshape(-1, 10)
            # _, gen = sess.run(get_generator(real_img_digit, noise_img, reuse=True),
            #                   feed_dict={noise_img: gen_sample, real_img_digit: labels})
            # if e % 1 == 0:
            #     gen = gen.reshape(-1, 28, 28)
            #     gen = (gen + 1) / 2
            #     save_genImages(gen, e)
        plot_loss(losses)
        # 保存模型
        saver.save(sess, 'checkpoints\\cgan.ckpt')


# 加载保存的模型
def test(is_save=False, name='cgan'):
    saver = tf.train.Saver(var_list=g_vars)
    with tf.Session() as sess:
        # saver.restore(sess, tf.train.latest_checkpoint("checkpoints"))
        saver.restore(sess, 'checkpoints\\cgan.ckpt')
        size=10
        sample_noise = get_noise(noise, size)
        #label = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0] * size
        label=np.eye(10)
        labels = np.array(label)
        labels = labels.reshape(-1, 10)
        _, gen_samples = sess.run(get_generator(real_img_digit, noise_img, reuse=True),
                                  feed_dict={noise_img: sample_noise, real_img_digit: labels})
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
            plt.savefig('cgan_pictures/' + name, dpi=100, format='png')
        else:
            plt.show()
        plt.close()


def use_model(is_save=False, times=1):
    for i in range(times):
        test(is_save, 'cgan_{}.png'.format(i+1))


if __name__ == '__main__':
    # train()
    use_model(is_save=True, times=10)
