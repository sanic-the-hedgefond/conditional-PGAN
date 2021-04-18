import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import os
from datetime import datetime
import yaml
from shutil import copyfile

from pcgan import PCGAN
from dataset import DatasetGenerator

continue_training = False

### ONLY IF CONTINUE TRAINING ###
#modeldir = 'C:/Users/Schnee/Desktop/MASTER/Training_Processes/pcgan/2021-04-18-070339/'
modeldir = 'training/2021-04-18-070339/'
ckptdir = modeldir + 'models/pcgan_stage_4_stabilize'
stage = 4
train_same_stage = True

### LOAD CONFIG ###
if continue_training:
  config_file = modeldir + 'config.yaml'
else:
  config_file = 'config.yaml'

with open(config_file) as f:
  config = yaml.load(f, Loader=yaml.FullLoader)

latent_dim = config['latent_dim']
num_chars = config['num_chars']
num_style_labels = config['num_style_labels']
step = config['step']
first_n_fonts = config['first_n_fonts']
batch_size = config['batch_size']
epochs = config['epochs']
discriminator_steps = config['discriminator_steps']
filters = config['filters']
font_dir = config['font_dir']
save_model = config['save_model']
###################

num_label_dim = num_chars
if num_style_labels > 0:
  num_label_dim += num_style_labels

training_dir = f'training/{datetime.now().strftime("%Y-%m-%d-%H%M%S")}/'

if not os.path.exists(f'{training_dir}images/models/'):
  os.makedirs(f'{training_dir}images/models/')
  os.makedirs(f'{training_dir}models/')

copyfile(config_file, f'{training_dir}config.yaml')

generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0004, beta_1=0.4, beta_2=0.99, epsilon=1e-8)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0004, beta_1=0.4, beta_2=0.99, epsilon=1e-8)

pcgan = PCGAN(
    latent_dim = latent_dim,
    num_classes = num_label_dim,
    filters = filters,
    d_steps = discriminator_steps,
)

def plot_models(name):
  tf.keras.utils.plot_model(pcgan.generator, to_file=f'{training_dir}images/models/generator_{pcgan.n_depth}_{name}.png', show_shapes=True)
  tf.keras.utils.plot_model(pcgan.discriminator, to_file=f'{training_dir}images/models/discriminator_{pcgan.n_depth}_{name}.png', show_shapes=True)

def train_stage(epochs, im_size, step, batch_size, name):
  training_set = DatasetGenerator(im_size=im_size, num_chars=num_chars, step=step, batch_size=batch_size, font_dir=font_dir, num_fonts=first_n_fonts, get_style_labels=(num_style_labels!=0))
  num_fonts = training_set.get_num_fonts()
  for cur_epoch in range(epochs): # Iterate epochs
    training_set.randomize_fonts()
    for cur_batch, batch in enumerate(training_set.batch): # Iterate batches
      pcgan.set_alpha((cur_batch+1)/(num_fonts//batch_size+1)/epochs + (cur_epoch)/epochs) # Set alpha for fade in layers (fade from 0 to 1 during whole stage)
      for cur_char in range(num_chars):
        batch_images, batch_labels = map(np.asarray, zip(*batch[cur_char::num_chars])) # Extract images and labels for current char from batch
        loss = pcgan.train_on_batch(x=batch_images, y=batch_labels, return_dict=True) # Train one batch
        print(f'{im_size}x{im_size} {name} // Epoch {cur_epoch+1}/{epochs} // Batch {cur_batch+1}/{num_fonts//batch_size+1} // Class {cur_char+1} // {loss}') # Logging
      pcgan.increment_random_seed()
      if save_model and cur_batch % 50 == 0:
        pcgan.generator.save(f'{training_dir}models/pcgan_tmp')
        pcgan.save_weights(f'{training_dir}models/pcgan_tmp')
    training_set.reset_generator()
    if save_model:
        pcgan.generator.save(f'{training_dir}models/pcgan_stage_{pcgan.n_depth}_{name}')
        pcgan.save_weights(f'{training_dir}models/pcgan_stage_{pcgan.n_depth}_{name}')

plot_models('init')

pcgan.compile(
    d_optimizer=discriminator_optimizer,
    g_optimizer=generator_optimizer,
)

if continue_training:
  for n_depth in range(1, stage+1):
    pcgan.n_depth = n_depth

    pcgan.fade_in_generator()
    pcgan.fade_in_discriminator()

    pcgan.compile(
        d_optimizer=discriminator_optimizer,
        g_optimizer=generator_optimizer,
    )

    pcgan.stabilize_generator()
    pcgan.stabilize_discriminator()

    pcgan.compile(
        d_optimizer=discriminator_optimizer,
        g_optimizer=generator_optimizer,
    )

  pcgan.load_weights(ckptdir)

  if train_same_stage:
    train_stage(epochs=epochs[n_depth], im_size=2**(n_depth+2), step=step[n_depth], batch_size=batch_size[n_depth], name='stabilize')

else:
  # Start training the initial generator and discriminator
  train_stage(epochs=epochs[0], im_size=4, step=step[0], batch_size=batch_size[0], name='init')
  stage = 1

# Train faded-in / stabilized generators and discriminators
for n_depth in range(stage+1, len(batch_size)):
  # Set current level(depth)
  pcgan.n_depth = n_depth

  # Put fade in generator and discriminator
  pcgan.fade_in_generator()
  pcgan.fade_in_discriminator()

  # Draw fade in generator and discriminator
  plot_models('fade_in')

  pcgan.compile(
      d_optimizer=discriminator_optimizer,
      g_optimizer=generator_optimizer,
  )

  train_stage(epochs=epochs[n_depth], im_size=2**(n_depth+2), step=step[n_depth], batch_size=batch_size[n_depth], name='fade_in')

  # Change to stabilized generator and discriminator
  pcgan.stabilize_generator()
  pcgan.stabilize_discriminator()

  # Draw stabilized generator and discriminator
  plot_models('stabilize')

  pcgan.compile(
      d_optimizer=discriminator_optimizer,
      g_optimizer=generator_optimizer,
  )

  train_stage(epochs=epochs[n_depth], im_size=2**(n_depth+2), step=step[n_depth], batch_size=batch_size[n_depth], name='stabilize')