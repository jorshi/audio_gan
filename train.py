#!/usr/bin/env python

"""
Audio GAN training script
"""

import os
import sys
import argparse
import json
from functools import partial
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from wpgan import WPGAN
from dcgan import DCGAN

BATCH_SIZE = 64
NUM_IMAGES = 4

# Possible models to train
models = {
    "wpgan": WPGAN,
    "dcgan": DCGAN
}


def load_dataset(filename, batch_size):
    """
    Load numpy object and convert to a TensorFlow dataset

    Args:
        filename (str): Location of numpy file
        batch_size (int): Size for each batch

    Returns:
        tf.Dataset
    """

    data = np.load(filename)
    data = data.reshape(data.shape[0], 16384, 1).astype('float32')

    # Normalize
    peak = max(abs(data.max()), abs(data.min()))
    data /= peak

    dataset = tf.data.Dataset.from_tensor_slices(data).shuffle(len(data)).batch(batch_size)
    return dataset


def save_images(test_batch, image_dir, image_prefix, model, epoch):

    predictions = model.generator(test_batch, training=False)
    fig = plt.figure(figsize=(2, 2))
    for i in range(predictions.shape[0]):
        plt.subplot(2, 2, i+1)
        plt.plot(predictions[i, :, 0])
        plt.axis('off')

    filename = os.path.join(image_dir, '{}_image_at_epoch_{:04d}.png'.format(image_prefix, epoch))
    plt.savefig(filename, dpi=150)
    plt.close(fig)


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
    parser.add_argument('-b', '--batch', default=64, help="Batch size", type=int)
    parser.add_argument('-e', '--epochs', default=50, help="Training epochs", type=int)
    parser.add_argument('-m', '--model', default="wpgan", help="Model type: {dcgan, wpgan}", type=str)
    parser.add_argument('-o', '--output', default=None, help="If set, save trained model to this file", type=str)
    parser.add_argument('-s', '--stats', default=None,
                        help="Save the loss/accuracy stats as JSON to this location", type=str)
    parser.add_argument('--ckpt_dir', default="training_checkpoints", help="Directory for training checkpoints", type=str)
    parser.add_argument('--ckpt_prefix', default="checkpoint", help="File prefix for checkpoint files", type=str)
    parser.add_argument('--ckpt_freq', default=0, help="How often to save checkpoints", type=int)
    parser.add_argument('--image_dir', default="training_images", help="Directory to save training images", type=str)
    parser.add_argument('--image_prefix', default=None,
                        help="File prefix to save image files (must be set to save images)", type=str)
    parser.add_argument('-u', '--upsample',
                        help="Generator upsample type: can set to resize or upsample, default is None",
                        default=None, type=str)
    parser.add_argument('-r', '--resume', help="Resume training from checkpoint",
                        action='store_const', const=True, default=False)
    parser.add_argument('--no_norm', dest='batch_norm', help="Turn off generator batch normalization",
                        action='store_const', const=False, default=True)
    parser.add_argument('-d', '--dropout', help="Generator Dropout", default=0.0, type=float)


    args = parser.parse_args(arguments)
    dataset = load_dataset(args.train_data, args.batch)

    if args.stats and not os.path.exists(os.path.dirname(args.stats)):
        raise Exception("Directory for train stats location does not exist: {}".format(args.stats))

    # Create model parameters
    kwargs = {
        'checkpoint_dir': args.ckpt_dir,
        'checkpoint_freq': args.ckpt_freq,
        'upsample': args.upsample,
        'batch_norm': args.batch_norm,
        'dropout': args.dropout
    }

    if args.ckpt_prefix:
        kwargs['checkpoint_prefix'] = args.ckpt_prefix

    # Create the model from the argument
    gan = models[args.model](**kwargs)

    if args.resume:
        gan.load_from_checkpoint(args.ckpt_dir)

    # Setup the callback to save images after each epoch
    callbacks = []
    if args.image_prefix is not None:
        if not os.path.exists(args.image_dir):
            os.makedirs(args.image_dir)

        # Create a latent space seed to share across images
        seed = tf.random.normal([NUM_IMAGES, gan.LATENT_SIZE])
        image_callback = partial(save_images, seed, args.image_dir, args.image_prefix)
        callbacks.append(image_callback)

    # Train the model
    history = gan.train(dataset, args.epochs, callbacks=callbacks)

    # Save the model
    if args.output is not None:
        gan.generator.save(args.output)

    # Save the training history
    if args.stats is not None:
        with open(args.stats, 'w') as fp:
            json.dump(history, fp, indent=True)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
