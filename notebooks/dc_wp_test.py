import sys
sys.path.append("../")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from dcgan_2 import DCGAN
from wpgan_2 import WPGAN

import IPython.display as ipd



snares = np.load("../data/snares.npy")
data = snares.reshape(snares.shape[0], 16384, 1).astype('float32')

# Normalize
peak = max(abs(data.max()), abs(data.min()))
data /= peak

dataset = tf.data.Dataset.from_tensor_slices(data).shuffle(len(data)).batch(64)


gan = DCGAN()
gan.train(dataset, epochs=3)
wpgan = WPGAN()
wpgan.train(dataset, epochs=3)


noise = tf.random.normal([1, 100])
generated_audio = wpgan.generator(noise, training=False)
audio = generated_audio[0,:,0]
plt.plot(audio)