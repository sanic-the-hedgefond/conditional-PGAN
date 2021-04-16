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

### LOAD CONFIG ###
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

copyfile(config_file, f'{training_dir}{config_file}')

generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.0, beta_2=0.99, epsilon=1e-8)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.0, beta_2=0.99, epsilon=1e-8)

pcgan = PCGAN(
    latent_dim = latent_dim,
    num_classes = num_label_dim,
    filters = filters,
    d_steps = discriminator_steps,
)

def generate_images(shape = (num_chars, 2), name='init', postfix='', seed=None):
  num_img = shape[0] * shape[1]

  random_latent_vectors = tf.random.normal(shape=[int(num_img/num_chars), latent_dim])
  if seed:
    random_latent_vectors = tf.random.normal(shape=[int(num_img/num_chars), latent_dim], seed=seed)
    postfix += f'_fixed_seed_{seed}'

  random_latent_vectors = tf.repeat(random_latent_vectors, num_chars, axis=0)

  labels = []
  for i in range(num_img):
    labels.append([0] * num_chars) 
    labels[i][i % num_chars] = 1
    if num_style_labels > 0:
      labels[i].extend(np.random.normal(loc=0.0, scale=0.2, size=num_style_labels).tolist())

  samples = []
  for i in range(shape[1]):
    index_start = i*num_chars
    index_end = min((i+1)*num_chars, num_img)
    samples.extend(pcgan.generator([random_latent_vectors[index_start:index_end], np.asarray(labels[index_start:index_end])]))
  samples = (np.asarray(samples) * 0.5) + 0.5

  img_size = samples.shape[1]

  '''
  imgs = []
  for i in range(num_img):
      imgs.append(tf.keras.preprocessing.image.array_to_img(samples[i]))
  '''

  num_rows = shape[1]

  imgs_alphabet  = []
  for i in range(num_rows):
    imgs_alphabet.append(cv2.hconcat(samples[i*num_chars:(i+1)*num_chars]))
  img_alphabets = cv2.vconcat(imgs_alphabet)
  title = f'{training_dir}images/plot_{img_size}x{img_size}_{name}{postfix}.png'
  plt.imsave(title, img_alphabets, cmap=plt.cm.gray)
  print(f'\n saved {title}')

  '''
  output_img = Image.new('L', (num_img//num_rows*img_size, img_size*num_rows))
  for i in range(len(imgs)):
      output_img.paste(imgs[i], (img_size*(i%(num_img//num_rows)), (i // (num_img//num_rows)) * img_size))

  output_img.save(title)
  '''

def plot_models(name):
  tf.keras.utils.plot_model(pcgan.generator, to_file=f'{training_dir}images/models/generator_{pcgan.n_depth}_{name}.png', show_shapes=True)
  tf.keras.utils.plot_model(pcgan.discriminator, to_file=f'{training_dir}images/models/discriminator_{pcgan.n_depth}_{name}.png', show_shapes=True)

def train_stage(epochs, im_size, step, batch_size, name):
  training_set = DatasetGenerator(im_size=im_size, num_chars=num_chars, step=step, batch_size=batch_size, font_dir=font_dir, num_fonts=first_n_fonts, get_style_labels=(num_style_labels!=0))
  num_fonts = training_set.get_num_fonts()
  for cur_epoch in range(epochs): # Iterate epochs
    training_set.randomize_fonts()
    for cur_batch, batch in enumerate(training_set.batch): # Iterate batches
      pcgan.set_alpha((cur_batch)/(num_fonts//batch_size+1)/epochs + (cur_epoch)/epochs) # Set alpha for fade in layers (fade from 0 to 1 during whole stage)
      for cur_char in range(num_chars):
        batch_images, batch_labels = map(np.asarray, zip(*batch[cur_char::num_chars])) # Extract images and labels for current char from batch
        loss = pcgan.train_on_batch(x=batch_images, y=batch_labels, return_dict=True) # Train one batch
        print(f'{im_size}x{im_size} {name} // Epoch {cur_epoch+1} // Batch {cur_batch+1}/{num_fonts//batch_size+1} // Class {cur_char+1} // {loss}') # Logging
      pcgan.increment_random_seed()
      if save_model and cur_batch % 50 == 0:
        pcgan.generator.save(f'{training_dir}models/pcgan_tmp')
        pcgan.save_weights(f'{training_dir}models/pcgan_tmp')
    training_set.reset_generator()
    #generate_images(name=name, postfix=f'_epoch{cur_epoch+1}')
    #generate_images(name=name, postfix=f'_epoch{cur_epoch+1}', seed=707)
  #generate_images(shape=(num_chars, 4), name=name, postfix='_final')
  #generate_images(shape=(num_chars, 4), name=name, postfix='_final', seed=707)
    if save_model:
        pcgan.generator.save(f'{training_dir}models/pcgan_stage_{pcgan.n_depth}_{name}')
        pcgan.save_weights(f'{training_dir}models/pcgan_stage_{pcgan.n_depth}_{name}')

plot_models('init')

pcgan.compile(
    d_optimizer=discriminator_optimizer,
    g_optimizer=generator_optimizer,
)

# Start training the initial generator and discriminator
train_stage(epochs=epochs[0], im_size=4, step=step[0], batch_size=batch_size[0], name='init')

# Train faded-in / stabilized generators and discriminators
for n_depth in range(1, len(batch_size)):
  # Set current level(depth)
  pcgan.n_depth = n_depth

  # Put fade in generator and discriminator
  pcgan.fade_in_generator()
  pcgan.fade_in_discriminator_new_embedding()

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