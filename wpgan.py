#!/usr/bin/env python
"""
Class containing code to train the WPGAN
"""

import os
from tqdm import tqdm
import tensorflow as tf
import wave_gan
import wave_gan_upsample
import wave_gan_resize


class WPGAN:

    LATENT_SIZE = 100

    def __init__(self, checkpoint_dir="./train_checkpoints", checkpoint_prefix="wpgan_ckpt", checkpoint_freq=0,
                 upsample=None):
        """
        Constructor
        """

        if upsample == "upsample":
            self.generator = wave_gan_upsample.make_generator_model(WPGAN.LATENT_SIZE)
            self.discriminator = wave_gan_upsample.make_discriminator_model()
        elif upsample == "resize":
            self.generator = wave_gan_resize.make_generator_model(WPGAN.LATENT_SIZE)
            self.discriminator = wave_gan_resize.make_discriminator_model()
        else:
            self.generator = wave_gan.make_generator_model(WPGAN.LATENT_SIZE)
            self.discriminator = wave_gan.make_discriminator_model(normalization=None)

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

        print(self.generator.summary())

    @staticmethod
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

    @staticmethod
    def generator_loss(fake_img):
        """
        Define the loss functions to be used for generator

        Args:
            fake_img (Tensor): result from discriminator

        Returns:
            Tensor: Loss
        """
        return -tf.reduce_mean(fake_img)

    @staticmethod
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
            noise = tf.random.normal([len(samples), WPGAN.LATENT_SIZE])
            with tf.GradientTape() as disc_tape:
                # Generate samples and get results from discriminator
                generated_samples = generator(noise, training=True)
                real_output = discriminator(samples, training=True)
                fake_output = discriminator(generated_samples, training=True)

                disc_loss = WPGAN.discriminator_loss(real_output, fake_output)

                # Gradient penalty
                gp = WPGAN.gradient_penalty(discriminator, len(samples), samples, generated_samples)

                # Weighted sum of losses
                total_loss = disc_loss + gp_weight * gp
                d_loss.append(float(total_loss))

                # Convert logits output from discriminator to prediction between 0 and 1
                real_pred = tf.round(tf.nn.sigmoid(real_output))
                fake_pred = tf.round(tf.nn.sigmoid(fake_output))

                # Calculate accuracy of predictions for real and fake outputs from discriminator
                real_accuracy.update_state(tf.ones_like(real_output), real_pred)
                fake_accuracy.update_state(tf.zeros_like(fake_output), fake_pred)

            gradients_of_discriminator = disc_tape.gradient(total_loss, discriminator.trainable_variables)

            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

        # Now train generator
        noise = tf.random.normal([len(samples), WPGAN.LATENT_SIZE])

        with tf.GradientTape() as gen_tape:
            generated_samples = generator(noise, training=True)
            fake_output = discriminator(generated_samples, training=True)
            gen_loss = WPGAN.generator_loss(fake_output)

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
            callbacks (list): A list functions to call after each epoch
        """

        for epoch in range(epochs):
            disc_loss, gen_loss = list(), list()

            for audio_batch in tqdm(dataset):
                d_loss, g_loss = WPGAN.train_step(audio_batch, self.generator, self.discriminator,
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
