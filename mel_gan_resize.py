#!/usr/bin/env python
"""
Functions for creating the WaveGan generator and discriminator, based on Donahue et al.
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine.input_spec import InputSpec


class Resize(tf.keras.layers.Layer):
    def __init__(self, size=2, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, **kwargs):
        super(Resize, self).__init__(**kwargs)
        self.size = int(size)
        self.input_spec = InputSpec(ndim=4)
        self.method = method

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        size = self.size * input_shape[1] if input_shape[1] is not None else None
        return tensor_shape.TensorShape([input_shape[0], size, size, input_shape[3]])

    def call(self, inputs):
        x = inputs
        x_shape = x.shape.as_list()
        x = tf.image.resize(x, [x_shape[1] * self.size, x_shape[2] * self.size], method=self.method)
        return x

    def get_config(self):
        config = {'size': self.size, 'method': self.method}
        base_config = super(Resize, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def make_generator_model(latent_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, normalization=True,
                         dropout=0.0):
    """
    Create the WaveGAN generator
    :return: Sequential Model
    """
    model = tf.keras.Sequential()
    model.add(layers.Dense(16 * 1024, use_bias=False, input_shape=(latent_size,)))
    if normalization:
        model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    if dropout > 0.0:
        model.add(layers.Dropout(dropout))

    model.add(layers.Reshape((4, 4, 1024)))
    assert model.output_shape == (None, 4, 4, 1024)  # Note: None is the batch size

    model.add(Resize(size=2, method=method))
    model.add(layers.Conv2D(512, 5, strides=1, padding='same', use_bias=False))
    assert model.output_shape == (None, 8, 8, 512)
    if normalization:
        model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    if dropout > 0.0:
        model.add(layers.Dropout(dropout))

    model.add(Resize(size=2, method=method))
    model.add(layers.Conv2D(256, 5, strides=1, padding='same', use_bias=False))
    assert model.output_shape == (None, 16, 16, 256)
    if normalization:
        model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    if dropout > 0.0:
        model.add(layers.Dropout(dropout))

    model.add(Resize(size=2, method=method))
    model.add(layers.Conv2D(128, 5, strides=1, padding='same', use_bias=False))
    assert model.output_shape == (None, 32, 32, 128)
    if normalization:
        model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    if dropout > 0.0:
        model.add(layers.Dropout(dropout))

    model.add(Resize(size=2, method=method))
    model.add(layers.Conv2D(64, 5, strides=1, padding='same', use_bias=False))
    assert model.output_shape == (None, 64, 64, 64)
    if normalization:
        model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    if dropout > 0.0:
        model.add(layers.Dropout(dropout))

    model.add(Resize(size=2, method=method))
    model.add(layers.Conv2D(1, 5, strides=1, padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 128, 128, 1)

    return model


def make_discriminator_model():
    """
    Create the WaveGAN discriminator
    :return: Sequential Model
    """
    model = tf.keras.Sequential()

    model.add(layers.Conv2D(64, 5, 2, padding='same', input_shape=[128, 128, 1]))
    assert (model.output_shape == (None, 64, 64, 64))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, 5, 2, padding='same'))
    assert (model.output_shape == (None, 32, 32, 128))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(256, 5, 2, padding='same'))
    assert (model.output_shape == (None, 16, 16, 256))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(512, 5, 2, padding='same'))
    assert (model.output_shape == (None, 8, 8, 512))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(1024, 5, 2, padding='same'))
    assert (model.output_shape == (None, 4, 4, 1024))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model
