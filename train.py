#!/usr/bin/env python

"""
Audio GAN training script
"""

import os
import sys
import argparse
import json
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
    parser.add_argument('-s', '--stats', default=None,
                        help="Save the loss/accuracy stats as JSON to this location", type=str)
    parser.add_argument('--ckpt_dir', default="training_checkpoints", help="Directory for training checkpoints", type=str)
    parser.add_argument('--ckpt_prefix', default=None, help="File prefix for checkpoint files", type=str)
    parser.add_argument('--ckpt_freq', default=0, help="How often to save checkpoints", type=int)

    args = parser.parse_args(arguments)
    dataset = load_dataset(args.train_data)

    if args.stats and not os.path.exists(os.path.dirname(args.stats)):
        raise Exception("Directory for train stats location does not exist: {}".format(args.stats))

    kwargs = {
        'checkpoint_dir': args.ckpt_dir,
        'checkpoint_freq': args.ckpt_freq
    }

    if args.ckpt_prefix:
        kwargs['checkpoint_prefix'] = args.ckpt_prefix

    gan = models[args.model](**kwargs)
    history = gan.train(dataset, args.epochs)

    # Save the model
    if args.output is not None:
        gan.generator.save(args.output)

    # Save the training history
    if args.stats is not None:
        with open(args.stats, 'w') as fp:
            json.dump(history, fp, indent=True)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
