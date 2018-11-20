# ---------------------------------------------------------
# Tensorflow WGAN-GP Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import tensorflow as tf
from solver import Solver

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('gpu_index', '0', 'gpu index if you have multiple gpus, default: 0')
tf.flags.DEFINE_integer('batch_size', 64, 'batch size, default: 64')
tf.flags.DEFINE_string('dataset', 'mnist', 'dataset name from [mnist, cifar10, imagenet64], default: mnist')

tf.flags.DEFINE_bool('is_train', True, 'training or inference mode, default: True')
tf.flags.DEFINE_float('learning_rate', 1e-4, 'initial learning rate for Adam, default: 0.0002')
tf.flags.DEFINE_integer('num_critic', 5, 'the number of iterations of the critic per generator iteration, default: 5')
tf.flags.DEFINE_integer('z_dim', 128, 'dimension of z vector, default: 128')
tf.flags.DEFINE_float('lambda_', 10., 'gradient penalty lambda hyperparameter, default: 10.')
tf.flags.DEFINE_float('beta1', 0.5, 'beta1 momentum term of Adam, default: 0.5')
tf.flags.DEFINE_float('beta2', 0.9, 'beta2 momentum term of Adam, default: 0.9')

tf.flags.DEFINE_integer('iters', 200000, 'number of iterations, default: 200000')
tf.flags.DEFINE_integer('print_freq', 100, 'print frequency for loss, default: 100')
tf.flags.DEFINE_integer('save_freq', 10000, 'save frequency for model, default: 10000')
tf.flags.DEFINE_integer('sample_freq', 500, 'sample frequency for saving image, default: 500')
tf.flags.DEFINE_integer('inception_freq', 1000, 'calculation frequence of inception score, default: 1000')
tf.flags.DEFINE_integer('sample_batch', 64, 'number of sampling images for check generator quality, default: 64')
tf.flags.DEFINE_string('load_model', None, 'folder of saved model taht you wish to continue training '
                       '(e.g. 20181017-1430), default: None')


def main(_):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_index

    solver = Solver(FLAGS)
    if FLAGS.is_train:
        solver.train()
    if not FLAGS.is_train:
        solver.test()


if __name__ == '__main__':
    tf.app.run()
