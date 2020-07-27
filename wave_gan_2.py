#!/usr/bin/env python
"""
Functions for creating the WaveGan generator and discriminator, based on Donahue et al.
"""

import tensorflow as tf
from tensorflow.keras import layers
#from keras.backend import tf as ktf


def make_generator_model(latent_size):
    """
    Create the WaveGAN generator
    :return: Sequential Model
    """
    model = tf.keras.Sequential()
    model.add(layers.Dense(16 * 1024, use_bias=False, input_shape=(latent_size,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((16, 1024)))
    assert model.output_shape == (None, 16, 1024)  # Note: None is the batch size
	
    model.add(layers.UpSampling1D(size=4))
    model.add(layers.Conv1D(512, 25, strides=1, padding='same', use_bias=False))
    assert model.output_shape == (None, 64, 512)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.UpSampling1D(size=4))
    model.add(layers.Conv1D(256, 25, strides=1, padding='same', use_bias=False))
    assert model.output_shape == (None, 256, 256)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.UpSampling1D(size=4))
    model.add(layers.Conv1D(128, 25, strides=1, padding='same', use_bias=False))
    assert model.output_shape == (None, 1024, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.UpSampling1D(size=4))
    model.add(layers.Conv1D(64, 25, strides=1, padding='same', use_bias=False))
    assert model.output_shape == (None, 4096, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.UpSampling1D(size=4))
    model.add(layers.Conv1D(1, 25, strides=1, padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 16384, 1)

    return model


def make_discriminator_model():
    """
    Create the WaveGAN discriminator
    :return: Sequential Model
    """
    model = tf.keras.Sequential()

    model.add(layers.Conv1D(64, 5, 4, padding='same', input_shape=[16384, 1]))
    assert (model.output_shape == (None, 4096, 64))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv1D(128, 5, 4, padding='same'))
    assert (model.output_shape == (None, 1024, 128))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv1D(256, 5, 4, padding='same'))
    assert (model.output_shape == (None, 256, 256))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv1D(512, 5, 4, padding='same'))
    assert (model.output_shape == (None, 64, 512))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv1D(1024, 5, 4, padding='same'))
    assert (model.output_shape == (None, 16, 1024))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model
