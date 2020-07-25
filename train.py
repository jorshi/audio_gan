#!/usr/bin/env python

"""
Audio GAN training script
"""

import sys
import argparse
import time
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import wave_gan

BATCH_SIZE = 64


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


def discriminator_loss(real_img, fake_img):
    """
    Define the loss functions to be used for discriminator
    This should be (fake_loss - real_loss)
    We will add the gradient penalty later to this loss function

    Args:
        real_img (Tensor): batch of real examples
        fake_img (Tensor): batch of examples from the generator

    Returns:
        Tensor: Loss
    """
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    return fake_loss - real_loss


def generator_loss(fake_img):
    """
    Define the loss functions to be used for generator

    Args:
        fake_img (Tensor): result from discriminator

    Returns:
        Tensor: Loss
    """
    return -tf.reduce_mean(fake_img)


@tf.function
def gradient_penalty(discriminator, batch_size, real_sounds, fake_sounds):
    """
    This is the innovation provided by the WP-GAN
    Calculates the gradient penalty.
    This loss is calculated on an interpolated image and added to the discriminator loss.

    Args:
        discriminator: the discriminator model
        batch_size: size of current batch
        real_sounds: real examples from this batch
        fake_sounds: examples from generator for this batch

    Returns: the gradient penalty for this batch
    """

    # get the interpolated image
    alpha = tf.random.normal([batch_size, 1, 1], 0.0, 1.0)
    diff = fake_sounds - real_sounds
    interpolated = real_sounds + alpha * diff

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        # 1. Get the discriminator output for this interpolated image.
        pred = discriminator(interpolated, training=True)

    # 2. Calculate the gradients w.r.t to this interpolated image.
    grads = gp_tape.gradient(pred, [interpolated])[0]
    # 3. Calculate the norm of the gradients
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return gp


@tf.function
def train_step(samples, generator, discriminator, generator_optimizer, discriminator_optimizer):
    """
    Update model weights on a single batch

    Args:
        samples: a batch of real examples
        generator:  the generator model
        discriminator: the discriminator model
        generator_optimizer: optimizer used for the generator
        discriminator_optimizer: optimizer used for the discriminator
    """

    gp_weight = 10.0
    disc_steps = 4

    d_loss = float()

    # Train discriminator
    for i in range(disc_steps):
        noise = tf.random.normal([len(samples), wave_gan.LATENT_SIZE])
        with tf.GradientTape() as disc_tape:
            generated_samples = generator(noise, training=True)

            real_output = discriminator(samples, training=True)
            fake_output = discriminator(generated_samples, training=True)

            disc_loss = discriminator_loss(real_output, fake_output)

            # Gradient penalty
            gp = gradient_penalty(discriminator, len(samples), samples, generated_samples)

            # Weighted sum of losses
            total_loss = disc_loss + gp_weight * gp
            d_loss = total_loss

        gradients_of_discriminator = disc_tape.gradient(total_loss, discriminator.trainable_variables)

        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    # Now train generator
    noise = tf.random.normal([len(samples), wave_gan.LATENT_SIZE])

    with tf.GradientTape() as gen_tape:
        generated_samples = generator(noise, training=True)

        real_output = discriminator(samples, training=True)
        fake_output = discriminator(generated_samples, training=True)

        gen_loss = generator_loss(fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    return d_loss, gen_loss


def train(dataset, epochs, output):
    """
    The training loop

    Args:
        dataset (tf.Dataset): the dataset to run training on
        epochs (int): number of training epochs to run
        output (str): if not None, then will save the trained model to this location
    """

    generator = wave_gan.make_generator_model()
    discriminator = wave_gan.make_discriminator_model()

    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    # Lists for storing loss history over training epochs
    history = {
        "disc_loss": list(),
        "gen_loss": list()
    }

    for epoch in range(epochs):
        start = time.time()

        for audio_batch in tqdm(dataset):
            disc_loss, gen_loss = train_step(audio_batch, generator, discriminator,
                                             generator_optimizer, discriminator_optimizer)

        print("Epoch {}, Discriminator Loss: {}, Generator Loss: {}".format(epoch, disc_loss, gen_loss))
        history["disc_loss"].append(disc_loss)
        history["gen_loss"].append(gen_loss)

    if output is not None:
        print("Saving model as {}".format(output))
        generator.save(output)

    return history


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
    parser.add_argument('-o', '--output', default=None, help="If set, save trained model to this file", type=str)

    args = parser.parse_args(arguments)
    dataset = load_dataset(args.train_data)
    train(dataset, args.epochs, args.output)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
