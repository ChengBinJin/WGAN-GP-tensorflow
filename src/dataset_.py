# ---------------------------------------------------------
# Tensorflow WGAN-GP Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import logging
import numpy as np
import scipy.misc
import tensorflow as tf

import utils as utils

logger = logging.getLogger(__name__)  # logger
logger.setLevel(logging.INFO)


def _init_logger(flags, log_path):
    if flags.is_train:
        formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
        # file handler
        file_handler = logging.FileHandler(os.path.join(log_path, 'dataset.log'))
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        # stream handler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        # add handlers
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)


class MnistDataset(object):
    def __init__(self, sess, flags, dataset_name):
        self.sess = sess
        self.flags = flags
        self.dataset_name = dataset_name
        self.image_size = (32, 32, 1)
        self.img_buffle = 100000  # image buffer for image shuflling
        self.num_trains, self.num_tests = 0, 0

        self.mnist_path = os.path.join('../../Data', self.dataset_name)
        self._load_mnist()

    def _load_mnist(self):
        logger.info('Load {} dataset...'.format(self.dataset_name))
        self.train_data, self.test_data = tf.keras.datasets.mnist.load_data()
        # self.train_data is tuple: (image, label)
        self.num_trains = self.train_data[0].shape[0]
        self.num_tests = self.test_data[0].shape[0]

        # TensorFlow Dataset API
        train_x, train_y = self.train_data
        dataset = tf.data.Dataset.from_tensor_slices(({'image': train_x}, train_y))
        dataset = dataset.shuffle(self.img_buffle).repeat().batch(self.flags.batch_size)

        iterator = dataset.make_one_shot_iterator()
        self.next_batch = iterator.get_next()

        logger.info('Load {} dataset SUCCESS!'.format(self.dataset_name))
        logger.info('Img size: {}'.format(self.image_size))
        logger.info('Num. of training data: {}'.format(self.num_trains))

    def train_next_batch(self, batch_size):
        batch_data = self.sess.run(self.next_batch)
        batch_imgs = batch_data[0]["image"]
        # batch_labels = batch_data[1]

        if self.flags.batch_size > batch_size:
            # reshape 784 vector to 28 x 28 x 1
            batch_imgs = np.reshape(batch_imgs[:batch_size], [batch_size, 28, 28])
        else:
            batch_imgs = np.reshape(batch_imgs, [self.flags.batch_size, 28, 28])

        imgs_32 = [scipy.misc.imresize(batch_imgs[idx], self.image_size[0:2])
                   for idx in range(batch_imgs.shape[0])]  # scipy.misc.imresize convert to uint8 type
        imgs_array = np.expand_dims(np.asarray(imgs_32).astype(np.float32), axis=3)

        return imgs_array / 127.5 - 1.  # from [0., 255.] to [-1., 1.]


class Cifar10(object):
    def __init__(self, flags, dataset_name):
        self.flags = flags
        self.dataset_name = dataset_name
        self.image_size = (32, 32, 3)
        self.num_trains = 0

        self.cifar10_path = os.path.join('../../Data', self.dataset_name)
        self._load_cifar10()

    def _load_cifar10(self):
        import cifar10

        cifar10.data_path = self.cifar10_path
        logger.info('Load {} dataset...'.format(self.dataset_name))

        # The CIFAR-10 data-set is about 163 MB and will be downloaded automatically if it is not
        # located in the given path.
        cifar10.maybe_download_and_extract()

        self.train_data, _, _ = cifar10.load_training_data()
        self.num_trains = self.train_data.shape[0]

        logger.info('Load {} dataset SUCCESS!'.format(self.dataset_name))
        logger.info('Img size: {}'.format(self.image_size))
        logger.info('Num. of training data: {}'.format(self.num_trains))

    def train_next_batch(self, batch_size):
        batch_imgs = self.train_data[np.random.choice(self.num_trains, batch_size, replace=False)]
        return batch_imgs * 2. - 1.  # from [0., 1.] to [-1., 1.]


class ImageNet64(object):
    def __init__(self, flags, dataset_name):
        self.flags = flags
        self.dataset_name = dataset_name
        self.image_size = (64, 64, 3)
        self.num_trains = 0

        self.imagenet64_path = os.path.join('../../Data', self.dataset_name, 'train_64x64')
        self._load_imagenet64()

    def _load_imagenet64(self):
        logger.info('Load {} dataset...'.format(self.dataset_name))
        self.train_data = utils.all_files_under(self.imagenet64_path, extension='.png')
        self.num_trains = len(self.train_data)

        logger.info('Load {} dataset SUCCESS!'.format(self.dataset_name))
        logger.info('Img size: {}'.format(self.image_size))
        logger.info('Num. of training data: {}'.format(self.num_trains))

    def train_next_batch(self, batch_size):
        batch_paths = np.random.choice(self.train_data, batch_size, replace=False)
        batch_imgs = [utils.load_data(batch_path, is_gray_scale=False) for batch_path in batch_paths]
        return np.asarray(batch_imgs)


# noinspection PyPep8Naming
def Dataset(sess, flags, dataset_name, log_path=None):
    if flags.is_train:
        _init_logger(flags, log_path)  # init logger

    if dataset_name == 'mnist':
        return MnistDataset(sess, flags, dataset_name)
    elif dataset_name == 'cifar10':
        return Cifar10(flags, dataset_name)
    elif dataset_name == 'imagenet64':
        return ImageNet64(flags, dataset_name)
    else:
        raise NotImplementedError
