# ---------------------------------------------------------
# Tensorflow WGAN-GP Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import logging
import numpy as np
import tensorflow as tf
from datetime import datetime

# noinspection PyPep8Naming
import plot as plot
from dataset_ import Dataset
from wgan_gp import WGAN_GP
from inception_score import get_inception_score

logger = logging.getLogger(__name__)  # logger
logger.setLevel(logging.INFO)


class Solver(object):
    def __init__(self, flags):
        run_config = tf.ConfigProto()
        run_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=run_config)

        self.flags = flags
        self.iter_time = 0
        self.num_examples_IS = 1000
        self._make_folders()
        self._init_logger()

        self.dataset = Dataset(self.sess, self.flags, self.flags.dataset, log_path=self.log_out_dir)
        self.model = WGAN_GP(self.sess, self.flags, self.dataset, log_path=self.log_out_dir)

        self.saver = tf.train.Saver()
        self.sess.run([tf.global_variables_initializer()])

        # tf_utils.show_all_variables()

    def _make_folders(self):
        if self.flags.is_train:  # train stage
            if self.flags.load_model is None:
                cur_time = datetime.now().strftime("%Y%m%d-%H%M")
                self.model_out_dir = "{}/model/{}".format(self.flags.dataset, cur_time)
                if not os.path.isdir(self.model_out_dir):
                    os.makedirs(self.model_out_dir)
            else:
                cur_time = self.flags.load_model
                self.model_out_dir = "{}/model/{}".format(self.flags.dataset, cur_time)

            self.sample_out_dir = "{}/sample/{}".format(self.flags.dataset, cur_time)
            if not os.path.isdir(self.sample_out_dir):
                os.makedirs(self.sample_out_dir)

            self.log_out_dir = "{}/logs/{}".format(self.flags.dataset, cur_time)
            self.train_writer = tf.summary.FileWriter("{}/logs/{}".format(self.flags.dataset, cur_time),
                                                      graph_def=self.sess.graph_def)

        elif not self.flags.is_train:  # test stage
            self.model_out_dir = "{}/model/{}".format(self.flags.dataset, self.flags.load_model)
            self.test_out_dir = "{}/test/{}".format(self.flags.dataset, self.flags.load_model)
            self.log_out_dir = "{}/logs/{}".format(self.flags.dataset, self.flags.load_model)

            if not os.path.isdir(self.test_out_dir):
                os.makedirs(self.test_out_dir)

    def _init_logger(self):
        formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
        # file handler
        file_handler = logging.FileHandler(os.path.join(self.log_out_dir, 'solver.log'))
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        # stream handler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        # add handlers
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

        if self.flags.is_train:
            logger.info('gpu_index: {}'.format(self.flags.gpu_index))
            logger.info('batch_size: {}'.format(self.flags.batch_size))
            logger.info('dataset: {}'.format(self.flags.dataset))

            logger.info('is_train: {}'.format(self.flags.is_train))
            logger.info('learning_rate: {}'.format(self.flags.learning_rate))
            logger.info('num_critic: {}'.format(self.flags.num_critic))
            logger.info('z_dim: {}'.format(self.flags.z_dim))
            logger.info('lambda_: {}'.format(self.flags.lambda_))
            logger.info('beta1: {}'.format(self.flags.beta1))
            logger.info('beta2: {}'.format(self.flags.beta2))

            logger.info('iters: {}'.format(self.flags.iters))
            logger.info('print_freq: {}'.format(self.flags.print_freq))
            logger.info('save_freq: {}'.format(self.flags.save_freq))
            logger.info('sample_freq: {}'.format(self.flags.sample_freq))
            logger.info('inception_freq: {}'.format(self.flags.inception_freq))
            logger.info('sample_batch: {}'.format(self.flags.sample_batch))
            logger.info('load_model: {}'.format(self.flags.load_model))

    def train(self):
        # load initialized checkpoint that provided
        if self.flags.load_model is not None:
            if self.load_model():
                logger.info(' [*] Load SUCCESS!\n')
            else:
                logger.info(' [!] Load Failed...\n')

        # for iter_time in range(self.flags.iters):
        while self.iter_time < self.flags.iters:
            # sampling images and save them
            self.sample(self.iter_time)

            # train_step
            loss, summary = self.model.train_step()
            self.model.print_info(loss, self.iter_time)
            self.train_writer.add_summary(summary, self.iter_time)
            self.train_writer.flush()

            if self.flags.dataset == 'cifar10':
                self.get_inception_score(self.iter_time)  # calculate inception score

            # save model
            self.save_model(self.iter_time)
            self.iter_time += 1

        self.save_model(self.flags.iters)

    def test(self):
        if self.load_model():
            logger.info(' [*] Load SUCCESS!')
        else:
            logger.info(' [!] Load Failed...')

        num_iters = 20
        for iter_time in range(num_iters):
            print('iter_time: {}'.format(iter_time))

            imgs = self.model.test_step()
            self.model.plots(imgs, iter_time, self.test_out_dir)

    def get_inception_score(self, iter_time):
        if np.mod(iter_time, self.flags.inception_freq) == 0:
            sample_size = 100
            all_samples = []
            for _ in range(int(self.num_examples_IS/sample_size)):
                imgs = self.model.sample_imgs(sample_size=sample_size)
                all_samples.append(imgs[0])

            all_samples = np.concatenate(all_samples, axis=0)
            all_samples = ((all_samples + 1.) * 255. / 2.).astype(np.uint8)

            mean_IS, std_IS = get_inception_score(list(all_samples), self.flags)
            # print('Inception score iter: {}, IS: {}'.format(self.iter_time, mean_IS))

            plot.plot('inception score', mean_IS)
            plot.flush(self.log_out_dir)  # write logs
            plot.tick()

    def sample(self, iter_time):
        if np.mod(iter_time, self.flags.sample_freq) == 0:
            imgs = self.model.sample_imgs(sample_size=self.flags.sample_batch)
            self.model.plots(imgs, iter_time, self.sample_out_dir)

    def save_model(self, iter_time):
        if np.mod(iter_time + 1, self.flags.save_freq) == 0:
            model_name = 'model'
            self.saver.save(self.sess, os.path.join(self.model_out_dir, model_name), global_step=iter_time)
            logger.info('[*] Model saved! Iter: {}'.format(iter_time))

    def load_model(self):
        logger.info(' [*] Reading checkpoint...')

        checkpoint = tf.train.get_checkpoint_state(self.model_out_dir)
        if checkpoint and checkpoint.model_checkpoint_path:
            ckpt_name = os.path.basename(checkpoint.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(self.model_out_dir, ckpt_name))

            meta_graph_path = checkpoint.model_checkpoint_path + '.meta'
            self.iter_time = int(meta_graph_path.split('-')[-1].split('.')[0])

            logger.info('[*] Load iter_time: {}'.format(self.iter_time))
            return True
        else:
            return False
