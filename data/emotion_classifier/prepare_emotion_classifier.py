"""
Example of Preprocessing pipeline for Emotion Classifier.

All required preprocessing is done beforehand, but you should
read this file to learn how preprocessing was done, and to
see how an embedding is computed for an example wavfile.
"""

import tensorflow as tf

import numpy as np
import pandas as pd

import soundfile as sf
from os.path import split, join

from vggish import vggish_input, vggish_params, vggish_postprocess, vggish_slim

import sys
sys.path.extend(['../..'])

from utils.utils import get_args
from utils.config import process_config


def create_vggish_network(sess, config):
    """
    Define VGGish model, load the checkpoint, and return a dictionary that points
    to the different tensors defined by the model.
    """
    vggish_slim.define_vggish_slim(training=False)
    vggish_params.EXAMPLE_HOP_SECONDS = config.vggish_hop_size
    
    vggish_slim.load_vggish_slim_checkpoint(
        sess, config.vggish_model_checkpoint_path)
    
    features_tensor = sess.graph.get_tensor_by_name(
        vggish_params.INPUT_TENSOR_NAME)
    embedding_tensor = sess.graph.get_tensor_by_name(
        vggish_params.OUTPUT_TENSOR_NAME)
    
    return {'features': features_tensor,
            'embedding': embedding_tensor}


def process_with_vggish(wavfile, vgg, sess, config):
    """
    Run the VGGish model, starting with a sound (x) at sample rate (sr).
    Return a whitened version of the embedding. Sound must be scaled to be
    floats between -1 and +1.
    """
    # Produce a batch of log mel spectrogram examples.
    input_features = vggish_input.wavfile_to_examples(wavfile)
    
    [embedding] = sess.run([vgg['embedding']],
        feed_dict={vgg['features']: input_features})
    
    # Postprocess the results to produce whitened quantized embeddings.
    pproc = vggish_postprocess.Postprocessor(config.vggish_pca_params_path)
    postprocessed_embedding = pproc.postprocess(embedding)
    
    return postprocessed_embedding


def preprocess_wavfile(wavfile, config):
    """
    Preprocess wavfile so that it is ready to be provided as input to VGGish network. 
    """
    # Read wavfile
    y, sr = sf.read(wavfile)
    if len(y.shape) == 2:
        y = y[:, 0].flatten()

    # Use rolling window to find where audio without silence starts and ends
    rolling = np.abs(pd.Series(y).rolling(config.win_size).sum()) \
        > config.silence_threshold
    
    # Cut off the silence
    nonzero = rolling.to_numpy().nonzero()
    start, end = nonzero[0][0], nonzero[0][-1]
    y = y[start : end + config.win_size]

    dir, filename = split(wavfile)

    # Save audio without silence for reference
    sf.write(join(dir, 'cut_' + filename), y, sr)

    # Pad with silence equally from both sides to reach the desired length
    pad_len = config.audio_len - y.size
    pad_left, pad_right = int(np.ceil(pad_len / 2.)), int(pad_len / 2.)
    
    y = np.concatenate((
                        np.zeros(pad_left),
                        y,
                        np.zeros(pad_right)
                    ))
    
    # Save centered audio padded with silence
    sf.write(join(dir, 'preprocessed_' + filename), y, sr)


def main(config):
    """
    Preprocessing pipeline for an example wavfile.
    All functions used here are defined and described above.
    """
    dir, filename = split(config.path_to_example)
    preprocess_wavfile(join(dir, filename), config)

    tf.reset_default_graph()
    sess = tf.Session()

    vgg = create_vggish_network(sess, config)

    embedding = process_with_vggish(
        join(dir, 'preprocessed_' + filename), vgg, sess, config)

    np.save(join(dir, filename.replace('.wav', '_embedding.npy')), embedding)


if __name__ == '__main__':
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)
        main(config)

    except Exception as e:
        print('Missing or invalid arguments %s' % e)
