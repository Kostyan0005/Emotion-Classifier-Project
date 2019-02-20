"""
Input pipeline for Emotion Classifier.

Load precomputed embeddings and labels for training,
development and testing sets. Then, build dataset API
and provide dataset initialization for different modes.
"""

import tensorflow as tf
import numpy as np

import sys
sys.path.extend(['..'])

from utils.utils import get_args
from utils.config import process_config


class EmotionClassifierDataLoader:
    def __init__(self, config):
        self.config = config

        # Load precomputed embeddings and labels
        self.train_embeddings = np.load(self.config.data_dir + 'train/train_embeddings.npy')
        self.dev_embeddings = np.load(self.config.data_dir + 'dev/dev_embeddings.npy')
        self.test_embeddings = np.load(self.config.data_dir + 'test/test_embeddings.npy')
        self.train_labels = np.load(self.config.data_dir + 'train/train_labels.npy')
        self.dev_labels = np.load(self.config.data_dir + 'dev/dev_labels.npy')
        self.test_labels = np.load(self.config.data_dir + 'test/test_labels.npy')

        # Check lens
        assert self.train_embeddings.shape[0] == self.train_labels.shape[0], \
        "Train filenames and labels should have same length"
        assert self.dev_embeddings.shape[0]  == self.dev_labels.shape[0], \
        "Dev filenames and labels should have same length"
        assert self.test_embeddings.shape[0]  == self.test_labels.shape[0], \
        "Test filenames and labels should have same length"

        # Define datasets sizes
        self.train_size = self.train_embeddings.shape[0]
        self.dev_size = self.dev_embeddings.shape[0]
        self.test_size = self.test_embeddings.shape[0]

        # Define number of iterations per epoch
        self.num_iterations_train = (self.train_size + self.config.batch_size - 1) \
            // self.config.batch_size
        self.num_iterations_dev  = (self.dev_size  + self.config.batch_size - 1) \
            // self.config.batch_size
        self.num_iterations_test  = (self.test_size  + self.config.batch_size - 1) \
            // self.config.batch_size

        self.features_placeholder = None
        self.labels_placeholder = None

        self.dataset = None
        self.iterator = None
        self.init_iterator_op = None
        self.next_batch = None

        self._build_dataset_api()


    def _build_dataset_api(self):
        with tf.device('/cpu:0'):
            self.features_placeholder = tf.placeholder(
                tf.float32,
                [None, self.config.audio_split_size,
                self.config.vggish_embedding_size])
            self.labels_placeholder = tf.placeholder(tf.int32, [None, ])
            self.mode_placeholder = tf.placeholder(tf.string, shape=())

            # Create a Dataset serving batches of images and labels
            # We don't repeat for multiple epochs because we always train and evaluate for one epoch
            self.dataset = (tf.data.Dataset.from_tensor_slices(
                    (self.features_placeholder, self.labels_placeholder)
                )
                .batch(self.config.batch_size)
                .prefetch(1)  # make sure you always have one batch ready to serve
            )

            # Create reinitializable iterator from dataset
            self.iterator = self.dataset.make_initializable_iterator()

            self.iterator_init_op = self.iterator.initializer

            self.next_batch = self.iterator.get_next()


    # There are 3 possible modes: 'train', 'dev', 'test'
    def initialize(self, sess, mode='train'):
        if mode == 'train':
            sess.run(self.iterator_init_op, feed_dict={
                self.features_placeholder: self.train_embeddings,
                self.labels_placeholder: self.train_labels,
                self.mode_placeholder: mode})
        elif mode == 'dev':
            sess.run(self.iterator_init_op, feed_dict={
                self.features_placeholder: self.dev_embeddings,
                self.labels_placeholder: self.dev_labels,
                self.mode_placeholder: mode})
        elif mode == 'test':
            sess.run(self.iterator_init_op, feed_dict={
                self.features_placeholder: self.test_embeddings,
                self.labels_placeholder: self.test_labels,
                self.mode_placeholder: mode})


    def get_inputs(self):
        return self.next_batch


def main(config):
    """
    Function to test from console, accepts config object as an argument
    """
    tf.reset_default_graph()
    sess = tf.Session()

    data_loader = EmotionClassifierDataLoader(config)

    embeddings, labels = data_loader.get_inputs()

    print('Train')
    data_loader.initialize(sess, mode='train')

    out_em, out_l = sess.run([embeddings, labels])

    print(out_em.shape, out_em.dtype)
    print(out_l.shape, out_l.dtype)

    print('Dev')
    data_loader.initialize(sess, mode='dev')

    out_em, out_l = sess.run([embeddings, labels])

    print(out_em.shape, out_em.dtype)
    print(out_l.shape, out_l.dtype)

    print('Test')
    data_loader.initialize(sess, mode='test')

    out_em, out_l = sess.run([embeddings, labels])

    print(out_em.shape, out_em.dtype)
    print(out_l.shape, out_l.dtype)


if __name__ == '__main__':
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)
        main(config)

    except Exception as e:
        print('Missing or invalid arguments %s' % e)
