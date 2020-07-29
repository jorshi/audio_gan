#!/usr/bin/env python
"""
Class containing code to train the 1D DCGAN
"""

import os
from tqdm import tqdm
import tensorflow as tf
import wave_gan
import wave_gan_upsample


class DCGAN:

    LATENT_SIZE = 100
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def __init__(self, checkpoint_dir="./train_checkpoints", checkpoint_prefix="dcgan_ckpt", checkpoint_freq=0,
                 upsample=False):
        """
        Constructor
        """

        if upsample:
            self.generator = wave_gan_upsample.make_generator_model(WPGAN.LATENT_SIZE)
            self.discriminator = wave_gan_upsample.make_discriminator_model()
        else:
            self.generator = wave_gan.make_generator_model(WPGAN.LATENT_SIZE)
            self.discriminator = wave_gan.make_discriminator_model(normalization=None)

        self.generator = wave_gan.make_generator_model(DCGAN.LATENT_SIZE)
        self.discriminator = wave_gan.make_discriminator_model()
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.real_accuracy = tf.keras.metrics.BinaryAccuracy()
        self.fake_accuracy = tf.keras.metrics.BinaryAccuracy()
        self.history = {
            "disc_loss": list(),
            "gen_loss": list(),
            "disc_real_accuracy": list(),
            "disc_fake_accuracy": list()
        }
        self.checkpoint_prefix = os.path.join(checkpoint_dir, checkpoint_prefix)
        self.checkpoint_frequency = checkpoint_freq
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                              discriminator_optimizer=self.discriminator_optimizer,
                                              generator=self.generator,
                                              discriminator=self.discriminator)

    def load_from_checkpoint(self, checkpoint_dir):
        """
        Reload model from checkpoint
        """
        self.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    @staticmethod
    def discriminator_loss(real, fake):
        """
        Define the loss functions to be used for discriminator
        This should be (fake_loss - real_loss)
        We will add the gradient penalty later to this loss function

        Args:
            real (Tensor): batch of real examples
            fake (Tensor): batch of examples from the generator

        Returns:
            Tensor: Loss
        """
        real_loss = DCGAN.cross_entropy(tf.ones_like(real), real)
        fake_loss = DCGAN.cross_entropy(tf.zeros_like(fake), fake)
        total_loss = real_loss + fake_loss
        return total_loss

    @staticmethod
    def generator_loss(fake):
        """
        Define the loss functions to be used for generator

        Args:
            fake (Tensor): result from discriminator

        Returns:
            Tensor: Loss
        """
        return DCGAN.cross_entropy(tf.ones_like(fake), fake)

    @staticmethod
    @tf.function
    def train_step(samples, generator, discriminator, generator_optimizer, discriminator_optimizer,
                   real_accuracy, fake_accuracy):
        """
        Update model weights on a single batch

        Args:
            samples: a batch of real examples
            generator:  the generator model
            discriminator: the discriminator model
            generator_optimizer: optimizer used for the generator
            discriminator_optimizer: optimizer used for the discriminator
            real_accuracy: binary accuracy metric for real samples
            fake_accuracy: binary accuracy metric for fake samples from generator
        """

        gp_weight = 10.0
        disc_steps = 4
        d_loss = list()

        # Train discriminator
        for i in range(disc_steps):
            noise = tf.random.normal([len(samples), DCGAN.LATENT_SIZE])
            with tf.GradientTape() as disc_tape:
                generated_samples = generator(noise, training=True)

                real_output = discriminator(samples, training=True)
                fake_output = discriminator(generated_samples, training=True)

                disc_loss = DCGAN.discriminator_loss(real_output, fake_output)
                d_loss.append(float(disc_loss))

                # Convert logits output from discriminator to prediction between 0 and 1
                real_pred = tf.round(tf.nn.sigmoid(real_output))
                fake_pred = tf.round(tf.nn.sigmoid(fake_output))

                # Calculate accuracy of predictions for real and fake outputs from discriminator
                real_accuracy.update_state(tf.ones_like(real_output), real_pred)
                fake_accuracy.update_state(tf.zeros_like(fake_output), fake_pred)

            gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

        # Now train generator
        noise = tf.random.normal([len(samples), DCGAN.LATENT_SIZE])

        with tf.GradientTape() as gen_tape:
            generated_samples = generator(noise, training=True)
            fake_output = discriminator(generated_samples, training=True)
            gen_loss = DCGAN.generator_loss(fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

        disc_loss = sum(d_loss) / disc_steps
        return disc_loss, float(gen_loss)

    def train(self, dataset, epochs, callbacks=None):
        """
        The training loop

        Args:
            dataset (tf.Dataset): the dataset to run training on
            epochs (int): number of training epochs to run
            callbacks (list): a list of callbacks to call after each epoch
        """

        for epoch in range(epochs):
            disc_loss, gen_loss = list(), list()

            for audio_batch in tqdm(dataset):
                d_loss, g_loss = DCGAN.train_step(audio_batch, self.generator, self.discriminator,
                                                  self.generator_optimizer, self.discriminator_optimizer,
                                                  self.real_accuracy, self.fake_accuracy)

                # Save the batch loss for generator and discriminator
                disc_loss.append(d_loss)
                gen_loss.append(g_loss)

            # Get accuracy scores for this epoch
            r_accuracy = self.real_accuracy.result()
            f_accuracy = self.fake_accuracy.result()

            # Average loss values
            d_loss = sum(disc_loss) / len(audio_batch)
            g_loss = sum(gen_loss) / len(audio_batch)

            print("Epoch {}, Discriminator Loss: {}, Generator Loss: {}".format(epoch, d_loss, g_loss))
            print("Epoch {}, Real accuracy: {}, Fake accuracy: {}".format(epoch, r_accuracy, f_accuracy))

            # Store loss and accuracy in history dictionary
            self.history["disc_loss"].append(float(d_loss))
            self.history["gen_loss"].append(float(g_loss))
            self.history["disc_real_accuracy"].append(float(r_accuracy))
            self.history["disc_fake_accuracy"].append(float(f_accuracy))

            # Reset accuracy scores for next epoch
            self.real_accuracy.reset_states()
            self.fake_accuracy.reset_states()

            # Save a checkpoint
            if self.checkpoint_frequency and (epoch + 1) % self.checkpoint_frequency == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)

            # Call callback functions
            if callbacks is not None:
                for callback in callbacks:
                    callback(self, epoch)

        return self.history
