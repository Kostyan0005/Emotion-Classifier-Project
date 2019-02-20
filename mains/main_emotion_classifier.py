"""
Starting point for Emotion Classifier project.
"""

import tensorflow as tf

import sys
sys.path.extend(['..'])

from data_generators.generator_emotion_classifier import EmotionClassifierDataLoader
from models.model_emotion_classifier import EmotionClassifierModel
from trainers.trainer_emotion_classifier import EmotionClassifierTrainer

from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import DefinedSummarizer
from utils.utils import get_args


def main():
    # capture the config path from the run arguments
    # then process the json configration file
    try:
        args = get_args()
        config = process_config(args.config)

    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])

    # create tensorflow session and limit gpu usage
    conf = tf.ConfigProto()
    conf.gpu_options.allow_growth=True
    sess = tf.Session(config=conf)

    # create your data generator
    data_loader = EmotionClassifierDataLoader(config)

    # create instance of the model you want
    model = EmotionClassifierModel(data_loader, config)

    # create tensorboard logger
    logger = DefinedSummarizer(sess, summary_dir=config.summary_dir,
                               scalar_tags=['train/loss_per_epoch', 'train/acc_per_epoch',
                                            'eval/loss_per_epoch','eval/acc_per_epoch'])

    # create trainer and pass all previous components to it
    trainer = EmotionClassifierTrainer(sess, model, config, logger, data_loader)


    # here you can train the model
    # trainer.train()

    # ... or evaluate its performance on development set
    # trainer.test()
    # trainer.test(mode='dev')

    # ... or test set
    trainer.test(mode='test')


if __name__ == '__main__':
    main()
