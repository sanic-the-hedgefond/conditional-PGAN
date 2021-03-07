import numpy as np
import tensorflow as tf

from matplotlib import pyplot
from math import sqrt
from PIL import Image
import os

from pgan import PGAN

from dataset import DatasetGenerator

# DEFINE PARAMETERS
latent_dim = 50
num_chars = 4
step = 10 # reduce size of dataset by 1/step
#font_dir = "../font_cgan/fonts/"
font_dir = "../../font_GAN/fonts/"

# Set the number of batches, epochs and steps for trainining.
batch_size = [32, 16, 16, 16, 8, 4, 4, 2, 2]
epochs = 3
discriminator_steps = 2
training_set = DatasetGenerator(im_size=4, num_chars=num_chars, step=step, batch_size=batch_size[0])
num_fonts = training_set.get_num_fonts()

# Instantiate the optimizer for both networks
# learning_rate will be equalized per each layers by the WeightScaling scheme
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.0, beta_2=0.99, epsilon=1e-8)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.0, beta_2=0.99, epsilon=1e-8)

# Instantiate the PGAN(PG-GAN) model.
pgan = PGAN(
    latent_dim = latent_dim,
    num_classes = num_chars,
    d_steps = discriminator_steps,
)

# Compile models
pgan.compile(
    d_optimizer=discriminator_optimizer,
    g_optimizer=generator_optimizer,
)

# Draw models
tf.keras.utils.plot_model(pgan.generator, to_file=f'images/generator_{pgan.n_depth}.png', show_shapes=True)
tf.keras.utils.plot_model(pgan.discriminator, to_file=f'images/discriminator_{pgan.n_depth}.png', show_shapes=True)

def generate_images(num_img = 16, name="init"):
  random_latent_vectors = tf.random.normal(shape=[num_img, latent_dim])

  labels = [0] * num_img
  for i in range(num_img):
    labels[i] = i % num_chars
  labels = np.asarray(labels)

  samples = pgan.generator([random_latent_vectors, labels])
  samples = (samples * 0.5) + 0.5
  n_grid = int(sqrt(num_img))

  im_size = samples.shape[1]

  fig, axes = pyplot.subplots(n_grid, n_grid, figsize=(4*n_grid, 4*n_grid))
  sample_grid = np.reshape(samples[:n_grid * n_grid], (n_grid, n_grid, samples.shape[1], samples.shape[2], samples.shape[3]))

  for i in range(n_grid):
    for j in range(n_grid):
      axes[i][j].set_axis_off()
      samples_grid_i_j = Image.fromarray((sample_grid[i][j] * 255).astype(np.uint8).squeeze(), mode="L")
      samples_grid_i_j = samples_grid_i_j.resize((128,128), resample=Image.NEAREST)
      axes[i][j].imshow(np.array(samples_grid_i_j), cmap='gray')
  title = f'images/plot_{im_size}x{im_size}_{name}.png'
  pyplot.savefig(title, bbox_inches='tight')
  print(f'\n saved {title}')
  pyplot.close(fig)

def train_stage(epochs=1, im_size=4, step=1, batch_size=32):
  training_set = DatasetGenerator(im_size=im_size, num_chars=num_chars, step=step, batch_size=batch_size)
  for i in range(epochs):
    for j, batch in enumerate(training_set.batch):
      pgan.set_alpha((j+1)/(num_fonts//batch_size)/epochs + (i)/epochs)
      for k in range(num_chars):
        batch_images, batch_labels = map(np.asarray, zip(*batch[j::num_chars]))
        loss = pgan.train_on_batch(x=batch_images, y=batch_labels, return_dict=True)
        print(f'Size {im_size}x{im_size} // Epoch {i+1} // Batch {j}/{num_fonts//batch_size} // Class {k} // {loss}')
      pgan.increment_random_seed()
    training_set.reset_generator()

# Start training the initial generator and discriminator
train_stage(epochs=epochs, im_size=4, step=step, batch_size=batch_size[0])
generate_images(num_img=32)

# Train faded-in / stabilized generators and discriminators
for n_depth in range(1, len(batch_size)):
  # Set current level(depth)
  pgan.n_depth = n_depth

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

  train_stage(epochs=epochs, im_size=2**(n_depth+2), step=step, batch_size=batch_size[n_depth])
  generate_images(num_img=32, name='fade_in')

  # Change to stabilized generator and discriminator
  pgan.stabilize_generator()
  pgan.stabilize_discriminator()

  # Draw stabilized generator and discriminator
  tf.keras.utils.plot_model(pgan.generator, to_file=f'images/generator_{n_depth}_stabilize.png', show_shapes=True)
  tf.keras.utils.plot_model(pgan.discriminator, to_file=f'images/discriminator_{n_depth}_stabilize.png', show_shapes=True)

  pgan.compile(
      d_optimizer=discriminator_optimizer,
      g_optimizer=generator_optimizer,
  )

  train_stage(epochs=epochs, im_size=2**(n_depth+2), step=step, batch_size=batch_size[n_depth])
  generate_images(num_img=32, name='stabilize')