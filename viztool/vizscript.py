import numpy as np
import tensorflow as tf
from PIL import Image
from string import ascii_uppercase, ascii_lowercase

text = 'BoldItalic'
save_dir = 'C:/Users/Schnee/Desktop/MASTER/Viz/20210322/'

modeldir = 'C:/Users/Schnee/Desktop/MASTER/Training_Processes/pcgan/2012-03-21-135259/pcgan_stage_5_stabilize/'
model = tf.saved_model.load(modeldir)
latent_dim = 20
chars = [c for c in ascii_uppercase + ascii_lowercase]

def get_random_latent():
    latent =  np.random.normal(0, 0.985, size=latent_dim).astype(np.float32)
    return np.tile(latent, (len(text), 1))

def get_input(text="default", latent=get_random_latent(), style=[0.5, 0.0, 0.0, 0.0, 0.8, 0.0]):
    input_latent = latent

    input_labels = []
    for char in text:
        label = [0.0] * len(chars)
        label[chars.index(char)] = 1.0
        label.extend(style)
        input_labels.append(label)
    input_labels = np.asarray(input_labels, dtype=np.float32)

    return [input_latent, input_labels]

def save_as_image(output, name="default.png"):
    img_size = output.shape[1]

    imgs = []
    for i in range(output.shape[0]):
        imgs.append(tf.keras.preprocessing.image.array_to_img(output[i]))

    output_img = Image.new('L', (img_size*len(text), img_size))
    for i in range(len(imgs)):
        output_img.paste(imgs[i], (img_size * i, 0))

    output_img.save(save_dir + name)

def get_interpolated_latents(num=2, steps=25):
    interpolated = []

    random_latent_01 = get_random_latent()

    for _ in range(num-1):
        random_latent_02 = get_random_latent()
        interpolated.extend(np.linspace(random_latent_01, random_latent_02, num=steps, axis=0))
        random_latent_01 = random_latent_02

    return interpolated

latents = get_interpolated_latents(5, 50)

for i, latent in enumerate(latents):
    input = get_input(text, latent=latent)
    output = model(input)
    save_as_image(output, f'{i:03d}_{text}.png')