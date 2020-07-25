#!/usr/bin/env python

"""
Audio GAN training script
"""

import sys
import argparse
import numpy as np
import tensorflow as tf
from wpgan import WPGAN
from dcgan import DCGAN

BATCH_SIZE = 64

# Possible models to train
models = {
    "wpgan": WPGAN,
    "dcgan": DCGAN
}


def load_dataset(filename):
    """
    Load numpy object and convert to a TensorFlow dataset

    Args:
        filename (str): Location of numpy file

    Returns:
        tf.Dataset
    """

    data = np.load(filename)
    data = data.reshape(data.shape[0], 16384, 1).astype('float32')

    # Normalize
    peak = max(abs(data.max()), abs(data.min()))
    data /= peak

    dataset = tf.data.Dataset.from_tensor_slices(data).shuffle(len(data)).batch(BATCH_SIZE)
    return dataset


def main(arguments):
    """
    Script entry point

    Args:
        arguments: command line arguments
    """

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('train_data', help="Location of .npy training data", type=str)
    parser.add_argument('-e', '--epochs', default=50, help="Training epochs", type=int)
    parser.add_argument('-m', '--model', default="wpgan", help="Model type: {dcgan, wpgan}", type=str)
    parser.add_argument('-o', '--output', default=None, help="If set, save trained model to this file", type=str)

    args = parser.parse_args(arguments)
    dataset = load_dataset(args.train_data)

    gan = models[args.model]()
    gan.train(dataset, args.epochs)

    if args.output is not None:
        gan.generator.save(args.output)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
