import numpy as np
import tensorflow as tf

from matplotlib import pyplot
from math import sqrt
from PIL import Image
import os
from datetime import datetime

from pgan import PGAN

from dataset import DatasetGenerator

# DEFINE PARAMETERS
latent_dim = 50
num_chars = 26
step = 1 # Reduce size of dataset by this factor
batch_size = [64, 32, 16, 16, 8, 4, 4, 2, 2]
epochs = 3
discriminator_steps = 4

training_dir = f'training/{datetime.now().strftime("%Y-%m-%d-%H%M%S")}/'
font_dir = '../Datasets/Fonts01CleanUp/' # Remote
#font_dir= 'C:/Users/Schnee/Datasets/Fonts01CleanUp/' # Local
image_dir = 'images/'

save_model = True

if not os.path.exists(f'{training_dir}{image_dir}models/'):
  os.makedirs(f'{training_dir}{image_dir}models/')

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

def generate_images(shape = (num_chars, 4), name='init', postfix='', seed=None):
  num_img = shape[0] * shape[1]

  random_latent_vectors = tf.random.normal(shape=[int(num_img/num_chars), latent_dim])
  if seed:
    random_latent_vectors = tf.random.normal(shape=[int(num_img/num_chars), latent_dim], seed=seed)
    postfix += f'_fixed_seed_{seed}'

  random_latent_vectors = tf.repeat(random_latent_vectors, num_chars, axis=0)

  labels = [0] * num_img
  for i in range(num_img):
    labels[i] = i % num_chars
  labels = np.asarray(labels)

  samples = pgan.generator([random_latent_vectors, labels])
  samples = (samples * 0.5) + 0.5

  im_size = samples.shape[1]

  fig, axes = pyplot.subplots(shape[1], shape[0], figsize=(4*shape[0], 4*shape[1]))
  sample_grid = np.reshape(samples[:shape[0] * shape[1]], (shape[0], shape[1], samples.shape[1], samples.shape[2], samples.shape[3]))

  for i in range(shape[1]):
    for j in range(shape[0]):
      axes[i][j].set_axis_off()
      samples_grid_i_j = Image.fromarray((sample_grid[j][i] * 255).astype(np.uint8).squeeze(), mode="L")
      samples_grid_i_j = samples_grid_i_j.resize((128,128), resample=Image.NEAREST)
      axes[i][j].imshow(np.array(samples_grid_i_j), cmap='gray')
  title = f'{training_dir}{image_dir}plot_{im_size}x{im_size}_{name}{postfix}.png'
  pyplot.savefig(title, bbox_inches='tight')
  print(f'\n saved {title}')
  pyplot.close(fig)

def plot_models():
  tf.keras.utils.plot_model(pgan.generator, to_file=f'{training_dir}{image_dir}models/generator_{pgan.n_depth}.png', show_shapes=True)
  tf.keras.utils.plot_model(pgan.discriminator, to_file=f'{training_dir}{image_dir}models/discriminator_{pgan.n_depth}.png', show_shapes=True)

def train_stage(epochs, im_size, step, batch_size, name):
  training_set = DatasetGenerator(im_size=im_size, num_chars=num_chars, step=step, batch_size=batch_size, font_dir=font_dir)
  num_fonts = training_set.get_num_fonts()
  for cur_epoch in range(epochs):
    for cur_batch, batch in enumerate(training_set.batch):
      pgan.set_alpha((cur_batch+1)/(num_fonts//batch_size)/epochs + (cur_epoch)/epochs)
      for cur_char in range(num_chars):
        batch_images, batch_labels = map(np.asarray, zip(*batch[cur_char::num_chars]))
        loss = pgan.train_on_batch(x=batch_images, y=batch_labels, return_dict=True)
        print(f'{im_size}x{im_size} {name} // Epoch {cur_epoch+1} // Batch {cur_batch}/{num_fonts//batch_size} // Class {cur_char} // {loss}')
      pgan.increment_random_seed()
    training_set.reset_generator()
    generate_images(name=name, postfix=f'_epoch{cur_epoch+1}')
    generate_images(name=name, postfix=f'_epoch{cur_epoch+1}', seed=707)
  generate_images(shape=(num_chars, 8), name=name, postfix='_final')
  generate_images(shape=(num_chars, 8), name=name, postfix='_final', seed=707)
  if save_model:
    pgan.generator.save(f'{training_dir}pgan_stage_{pgan.n_depth}_{name}')

plot_models()

pgan.compile(
    d_optimizer=discriminator_optimizer,
    g_optimizer=generator_optimizer,
)

# Start training the initial generator and discriminator
train_stage(epochs=epochs, im_size=4, step=step, batch_size=batch_size[0], name='init')

# Train faded-in / stabilized generators and discriminators
for n_depth in range(1, len(batch_size)):
  # Set current level(depth)
  pgan.n_depth = n_depth

  # Put fade in generator and discriminator
  pgan.fade_in_generator()
  pgan.fade_in_discriminator()

  # Draw fade in generator and discriminator
  plot_models()

  pgan.compile(
      d_optimizer=discriminator_optimizer,
      g_optimizer=generator_optimizer,
  )

  train_stage(epochs=epochs, im_size=2**(n_depth+2), step=step, batch_size=batch_size[n_depth], name='fade_in')

  # Change to stabilized generator and discriminator
  pgan.stabilize_generator()
  pgan.stabilize_discriminator()

  # Draw stabilized generator and discriminator
  plot_models()

  pgan.compile(
      d_optimizer=discriminator_optimizer,
      g_optimizer=generator_optimizer,
  )

  train_stage(epochs=epochs, im_size=2**(n_depth+2), step=step, batch_size=batch_size[n_depth], name='stabilize')