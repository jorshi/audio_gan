#!/usr/bin/env python

"""
Script for creating evaluation datasets
"""


import os
import sys
import argparse
import numpy as np
import tensorflow as tf

sys.path.append('../')
import wave_gan_resize

custom_objects = {
    "Resize": wave_gan_resize.Resize
}




def load_model(model):
    """
    Load a TensorFlow Keras model
    """
    generator = tf.keras.models.load_model(model, custom_objects=custom_objects)
    return generator

def generate_samples(model, num_samples, latent):
    """
    Generate a set of samples from model and return a numpy array
    """

    noise = tf.random.normal([num_samples, latent])
    samples = model(noise)[:, :, 0]
    return np.array(samples)



def main(arguments):

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('model', help="Model to use to render dataset", type=str)
    parser.add_argument('num_samples', help="Number of samples to generate from model", type=int)
    parser.add_argument('output', help="Output file", type=str)
    parser.add_argument('-l', '--latent', help="Size of the latent space", default=100, type=int)

    args = parser.parse_args(arguments)
    generator = load_model(args.model)
    samples = generate_samples(generator, args.num_samples, args.latent)
    np.save(args.output, samples)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))