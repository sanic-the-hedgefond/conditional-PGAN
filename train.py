import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from matplotlib import pyplot
from math import sqrt
from PIL import Image
import os

from pgan import PGAN, WeightedSum
from tensorflow.keras import backend

import dataset
import datasetGenerator

# Create a Keras callback that periodically saves generated images and updates alpha in WeightedSum layers
class GANMonitor(keras.callbacks.Callback):
  def __init__(self, num_img=16, latent_dim=512, prefix=''):
    self.num_img = num_img
    self.latent_dim = latent_dim

    random_latent_vectors = tf.random.normal(shape=[num_img, self.latent_dim])

    self.random_latent_vectors = random_latent_vectors
    self.steps_per_epoch = 0
    self.epochs = 0
    self.steps = self.steps_per_epoch * self.epochs
    self.n_epoch = 0
    self.prefix = prefix
  
  def set_prefix(self, prefix=''):
    self.prefix = prefix
  
  def set_steps(self, steps_per_epoch, epochs):
    self.steps_per_epoch = steps_per_epoch
    self.epochs = epochs
    self.steps = self.steps_per_epoch * self.epochs

  def on_epoch_begin(self, epoch, logs=None):
    self.n_epoch = epoch

  def on_epoch_end(self, epoch, logs=None):
    self.random_latent_vectors = tf.random.normal(shape=[self.num_img, self.latent_dim])

    labels = [0] * self.num_img
    for i in range(self.num_img):
      labels[i] = i % NUM_CHARS
    labels = np.asarray(labels)

    samples = self.model.generator([self.random_latent_vectors, labels])
    samples = (samples * 0.5) + 0.5
    n_grid = int(sqrt(self.num_img))

    fig, axes = pyplot.subplots(n_grid, n_grid, figsize=(4*n_grid, 4*n_grid))
    sample_grid = np.reshape(samples[:n_grid * n_grid], (n_grid, n_grid, samples.shape[1], samples.shape[2], samples.shape[3]))

    for i in range(n_grid):
      for j in range(n_grid):
        axes[i][j].set_axis_off()
        samples_grid_i_j = Image.fromarray((sample_grid[i][j] * 255).astype(np.uint8).squeeze(), mode="L")
        samples_grid_i_j = samples_grid_i_j.resize((128,128), resample=Image.NEAREST)
        axes[i][j].imshow(np.array(samples_grid_i_j), cmap='gray')
    title = f'images/plot_{self.prefix}_{epoch:05d}.png'
    pyplot.savefig(title, bbox_inches='tight')
    print(f'\n saved {title}')
    pyplot.close(fig)
  

  def on_batch_begin(self, batch, logs=None):
    # Update alpha in WeightedSum layers
    alpha = ((batch*2) + self.n_epoch * self.steps_per_epoch) / (self.steps + 1)
    backend.set_value(self.model.alpha, alpha)
    for layer in self.model.generator.layers:
      if isinstance(layer, WeightedSum):
        backend.set_value(layer.alpha, alpha)
    for layer in self.model.discriminator.layers:
      if isinstance(layer, WeightedSum):
        backend.set_value(layer.alpha, alpha)


# DEFINE PARAMETERS
NOISE_DIM = 50
NUM_CHARS = 4
STEP = 15 # reduce size of dataset by 1/STEP
#FONT_DIR = "../font_cgan/fonts/"
FONT_DIR = "../../font_GAN/fonts/"
#training_set = dataset.get_labeled_data(IM_SIZE=4, num_chars=NUM_CHARS, step=STEP, FONT_DIR=FONT_DIR)
training_set = datasetGenerator.data_generator(IM_SIZE=4, num_chars=NUM_CHARS, step=STEP, FONT_DIR=FONT_DIR)

# Set the number of batches, epochs and steps for trainining.
BATCH_SIZE = [32, 16, 16, 16, 8, 4, 4, 2, 2]
EPOCHS = 5
DISCRIMINATOR_STEPS = 2
NUM_FONTS = datasetGenerator.get_num_fonts(step=STEP)
STEPS_PER_EPOCH = NUM_FONTS * NUM_CHARS / BATCH_SIZE[0]

#print("Train IMG shape: ", next(iter(train_dataset))[0].shape)

# Instantiate the optimizer for both networks
# learning_rate will be equalized per each layers by the WeightScaling scheme
generator_optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.0, beta_2=0.99, epsilon=1e-8)
discriminator_optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.0, beta_2=0.99, epsilon=1e-8)

cbk = GANMonitor(num_img=64, latent_dim=NOISE_DIM, prefix='0_init')
cbk.set_steps(steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS)

# Instantiate the PGAN(PG-GAN) model.
pgan = PGAN(
    latent_dim = NOISE_DIM,
    num_classes = NUM_CHARS,
    d_steps = DISCRIMINATOR_STEPS,
)

checkpoint_path = f"ckpts/pgan_{cbk.prefix}.ckpt"

# Compile models
pgan.compile(
    d_optimizer=discriminator_optimizer,
    g_optimizer=generator_optimizer,
)

# Draw models
tf.keras.utils.plot_model(pgan.generator, to_file=f'images/generator_{pgan.n_depth}.png', show_shapes=True)
tf.keras.utils.plot_model(pgan.discriminator, to_file=f'images/discriminator_{pgan.n_depth}.png', show_shapes=True)

# Start training the initial generator and discriminator
pgan.fit(training_set, batch_size=BATCH_SIZE[0], epochs = EPOCHS, callbacks=[cbk])

# Train faded-in / stabilized generators and discriminators
for n_depth in range(1, len(BATCH_SIZE)):
  # Set current level(depth)
  pgan.n_depth = n_depth

  # Set parameters like epochs, steps, batch size and image size
  training_set = dataset.get_labeled_data(IM_SIZE=2**(n_depth+2), num_chars=NUM_CHARS, step=STEP, FONT_DIR=FONT_DIR)
  STEPS_PER_EPOCH = len(training_set[0][0]) / BATCH_SIZE[n_depth]
  
  cbk.set_steps(steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS)

  # Put fade in generator and discriminator
  pgan.fade_in_generator()
  pgan.fade_in_discriminator()

  # Draw fade in generator and discriminator
  tf.keras.utils.plot_model(pgan.generator, to_file=f'images/generator_{n_depth}_fade_in.png', show_shapes=True)
  tf.keras.utils.plot_model(pgan.discriminator, to_file=f'images/discriminator_{n_depth}_fade_in.png', show_shapes=True)

  pgan.compile(
      d_optimizer=discriminator_optimizer,
      g_optimizer=generator_optimizer,
  )
  for i in range(NUM_CHARS):
    cbk.set_prefix(f"fade_in_{2**(n_depth+2)}x{2**(n_depth+2)}_class_{i}")

    # Train fade in generator and discriminator
    pgan.fit(x=training_set[0][i], y=training_set[1][i], batch_size=BATCH_SIZE[n_depth], epochs = EPOCHS, callbacks=[cbk])
  # Save models
  checkpoint_path = f"ckpts/pgan_{cbk.prefix}.ckpt"
  #pgan.save_weights(checkpoint_path)

  # Change to stabilized generator and discriminator
  cbk.set_prefix(prefix=f'{n_depth}_stabilize')
  pgan.stabilize_generator()
  pgan.stabilize_discriminator()

  # Draw stabilized generator and discriminator
  tf.keras.utils.plot_model(pgan.generator, to_file=f'images/generator_{n_depth}_stabilize.png', show_shapes=True)
  tf.keras.utils.plot_model(pgan.discriminator, to_file=f'images/discriminator_{n_depth}_stabilize.png', show_shapes=True)
  pgan.compile(
      d_optimizer=discriminator_optimizer,
      g_optimizer=generator_optimizer,
  )

  for i in range(NUM_CHARS):
    cbk.set_prefix(f"stabilize_{2**(n_depth+2)}x{2**(n_depth+2)}_class_{i}")
    # Train stabilized generator and discriminator
    pgan.fit(x=training_set[0][i], y=training_set[1][i], batch_size=BATCH_SIZE[n_depth], epochs = EPOCHS, callbacks=[cbk])

  # Save models
  checkpoint_path = f"ckpts/pgan_{2**(n_depth+2)}x{2**(n_depth+2)}.ckpt"
  pgan.save_weights(checkpoint_path)