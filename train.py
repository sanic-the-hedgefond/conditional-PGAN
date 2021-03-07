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

from dataset import DatasetGenerator

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
      labels[i] = i % num_chars
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


# DEFINE PARAMETERS
latent_dim = 50
num_chars = 4
step = 5 # reduce size of dataset by 1/step
#font_dir = "../font_cgan/fonts/"
font_dir = "../../font_GAN/fonts/"
#training_set = dataset.get_labeled_data(IM_SIZE=4, num_chars=num_chars, step=step, font_dir=font_dir)
#training_set = datasetGenerator.data_generator(IM_SIZE=4, num_chars=num_chars, step=step, font_dir=font_dir)

# Set the number of batches, epochs and steps for trainining.
batch_size = [32, 16, 16, 16, 8, 4, 4, 2, 2]
epochs = 2
discriminator_steps = 2
training_set = DatasetGenerator(im_size=4, num_chars=num_chars, step=step, batch_size=batch_size[0])
num_fonts = training_set.get_num_fonts()
steps_per_epoch = num_fonts * num_chars / batch_size[0]

# Instantiate the optimizer for both networks
# learning_rate will be equalized per each layers by the WeightScaling scheme
generator_optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.0, beta_2=0.99, epsilon=1e-8)
discriminator_optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.0, beta_2=0.99, epsilon=1e-8)

cbk = GANMonitor(num_img=64, latent_dim=latent_dim, prefix='0_init')
cbk.set_steps(steps_per_epoch=steps_per_epoch, epochs=epochs)

# Instantiate the PGAN(PG-GAN) model.
pgan = PGAN(
    latent_dim = latent_dim,
    num_classes = num_chars,
    d_steps = discriminator_steps,
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
for i, batch in enumerate(training_set.batch):
  for j in range(num_chars):
    batch_images, batch_labels = map(np.asarray, zip(*batch[i::num_chars]))
    loss = pgan.train_on_batch(x=batch_images, y=batch_labels, return_dict=True)
    print(f'Batch {i}/{num_fonts//batch_size[0]} completed for class {j} with {loss}')
  pgan.increment_seed()
  pgan.set_alpha(i/(num_fonts//batch_size[0]))

# Train faded-in / stabilized generators and discriminators
for n_depth in range(1, len(batch_size)):
  # Set current level(depth)
  pgan.n_depth = n_depth

  # Set parameters like epochs, steps, batch size and image size
  training_set = dataset.get_labeled_data(IM_SIZE=2**(n_depth+2), num_chars=num_chars, step=step, font_dir=font_dir)
  steps_per_epoch = len(training_set[0][0]) / batch_size[n_depth]
  
  cbk.set_steps(steps_per_epoch=steps_per_epoch, epochs=epochs)

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
  for i in range(num_chars):
    cbk.set_prefix(f"fade_in_{2**(n_depth+2)}x{2**(n_depth+2)}_class_{i}")

    # Train fade in generator and discriminator
    pgan.fit(x=training_set[0][i], y=training_set[1][i], batch_size=batch_size[n_depth], epochs = epochs, callbacks=[cbk])
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

  for i in range(num_chars):
    cbk.set_prefix(f"stabilize_{2**(n_depth+2)}x{2**(n_depth+2)}_class_{i}")
    # Train stabilized generator and discriminator
    pgan.fit(x=training_set[0][i], y=training_set[1][i], batch_size=batch_size[n_depth], epochs = epochs, callbacks=[cbk])

  # Save models
  checkpoint_path = f"ckpts/pgan_{2**(n_depth+2)}x{2**(n_depth+2)}.ckpt"
  pgan.save_weights(checkpoint_path)