import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL

# Requires tensorflow==2.3.0rc0
import tensorflow as tf
from tensorflow.keras import layers
import keras.backend as K



import time
from tqdm import tqdm

import IPython.display as ipd


train_audio = np.load('./snares.npy')

train_audio = train_audio.reshape(train_audio.shape[0], 16384, 1).astype('float32')

# Normalize
peak = max(abs(train_audio.max()), abs(train_audio.min()))
train_audio /= peak

train_audio.shape

BUFFER_SIZE = 2210
BATCH_SIZE = 1


# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_audio).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)



def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(16*1024, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Reshape((16, 1024)))
    assert model.output_shape == (None, 16, 1024) # Note: None is the batch size
    
    model.add(layers.Conv1DTranspose(512, 25, strides=4, padding='same', use_bias=False))
    assert model.output_shape == (None, 64, 512)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Conv1DTranspose(256, 25, strides=4, padding='same', use_bias=False))
    assert model.output_shape == (None, 256, 256)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Conv1DTranspose(128, 25, strides=4, padding='same', use_bias=False))
    assert model.output_shape == (None, 1024, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Conv1DTranspose(64, 25, strides=4, padding='same', use_bias=False))
    assert model.output_shape == (None, 4096, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Conv1DTranspose(1, 25, strides=4, padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 16384, 1)

    return model
    
    
def make_discriminator_model():
    model = tf.keras.Sequential()
    
    model.add(layers.Conv1D(64, 5, 4, padding='same', input_shape=[16384, 1]))
    assert(model.output_shape == (None, 4096, 64))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Conv1D(128, 5, 4, padding='same'))
    assert(model.output_shape == (None, 1024, 128))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Conv1D(256, 5, 4, padding='same'))
    assert(model.output_shape == (None, 256, 256))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Conv1D(512, 5, 4, padding='same'))
    assert(model.output_shape == (None, 64, 512))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Conv1D(1024, 5, 4, padding='same'))
    assert(model.output_shape == (None, 16, 1024))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model
    
    
# Define the loss functions to be used for discrimiator
# This should be (fake_loss - real_loss)
# We will add the gradient penalty later to this loss function
def discriminator_loss(real_img, fake_img):
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    return fake_loss - real_loss

    

# Define the loss functions to be used for generator
def generator_loss(fake_img):
    return -tf.reduce_mean(fake_img)

    
generator = make_generator_model()
#generator.summary()

noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)
audio = generated_image[0,:,0]
plt.plot(audio)


ipd.Audio(audio, rate=16000)


discriminator = make_discriminator_model()
decision = discriminator(generated_image)
print (decision)


discriminator_loss(generated_image, generated_image)


generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


checkpoint_dir = './training_checkpoints_audio_gan'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
                                 
                                 
EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 4

# We will reuse this seed over time (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])


# This is the innovation provided by the WP-GAN
@tf.function
def gradient_penalty(discriminator, batch_size, real_sounds, fake_sounds):
    #Calculates the gradient penalty.
    #This loss is calculated on an interpolated image and added to the discriminator loss.
    
    
    # get the interplated image
    alpha = tf.random.normal([batch_size, 1, 1], 0.0, 1.0)
    diff = fake_sounds - real_sounds
    interpolated = real_sounds + alpha * diff

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        # 1. Get the discriminator output for this interpolated image.
        pred = discriminator(interpolated, training=True)

    # 2. Calculate the gradients w.r.t to this interpolated image.
    grads = gp_tape.gradient(pred, [interpolated])[0]
    # 3. Calcuate the norm of the gradients
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return gp


# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(samples):
    gp_weight=10.0
    disc_steps=4

    # Train discriminator
    for i in range(disc_steps):
        noise = tf.random.normal([BATCH_SIZE, noise_dim])
        with tf.GradientTape() as disc_tape:
            generated_samples = generator(noise, training=True)

            real_output = discriminator(samples, training=True)
            fake_output = discriminator(generated_samples, training=True)

            disc_loss = discriminator_loss(real_output, fake_output)
        
            # Gradient penalty
            gp = gradient_penalty(discriminator, BATCH_SIZE, samples, generated_samples)

            # Weighted sum of losses
            total_loss=disc_loss+gp_weight*gp

        gradients_of_discriminator = disc_tape.gradient(total_loss, discriminator.trainable_variables)

        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    # Now train generator
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape:
        generated_samples = generator(noise, training=True)

        real_output = discriminator(samples, training=True)
        fake_output = discriminator(generated_samples, training=True)

        gen_loss = generator_loss(fake_output)

        
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))


    
    
def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    for audio_batch in tqdm(dataset):
      train_step(audio_batch)

    # Produce images for the GIF as we go
    ipd.clear_output(wait=True)
    generate_and_save_audio_images(generator,
                                   epoch + 1,
                                   seed)

    # Save the model every 15 epochs
    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # Generate after the final epoch
  ipd.clear_output(wait=True)
  generate_and_save_audio_images(generator,
                                 epochs,
                                 seed)
                                 
                                 
def generate_and_save_audio_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(2,2))

  for i in range(predictions.shape[0]):
      plt.subplot(2, 2, i+1)
      plt.plot(predictions[i, :, 0])
      plt.axis('off')

  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()
  
  
train(train_dataset, EPOCHS)



generator.save('snare_drum_generator.h5')

noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)
audio = generated_image[0,:,0]
plt.plot(audio)


ipd.Audio(audio, rate=16000)