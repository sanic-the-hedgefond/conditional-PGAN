import numpy as np
import tensorflow as tf
from PIL import Image
from string import ascii_uppercase, ascii_lowercase
import os
import yaml

text = 'alphabet'
save_dir = 'C:/Users/Schnee/Desktop/MASTER/Viz/20210420_07_alphabet128_25steps/'

#modeldir = 'C:/Users/Schnee/Desktop/MASTER/Training_Processes/pcgan/2012-03-21-135259/pcgan_stage_6_fade_in/'
#modeldir = 'C:/Users/Schnee/Desktop/MASTER/Training_Processes/pcgan/2021-04-02-225335/models/pcgan_stage_5_stabilize/'
modeldir = 'C:/Users/Schnee/Desktop/MASTER/Training_Processes/pcgan/2021-04-14-073102/models/pcgan_tmp/'
#modeldir = 'C:/Users/Schnee/Desktop/MASTER/Training_Processes/pcgan/2021-04-19-150642/models/pcgan_stage7_200batches/'

latentdir = 'C:/Users/Schnee/Desktop/MASTER/Training_Processes/pcgan/2021-04-14-073102/models/viztool/'
#latentdir = 'C:/Users/Schnee/Desktop/MASTER/Training_Processes/pcgan/2021-04-19-150642/models/viztool/'

steps = 25

'''
latentfiles =   ['2021-04-19-171604.yaml', '2021-04-19-171555.yaml', '2021-04-19-171546.yaml', '2021-04-19-173316.yaml', '2021-04-19-172128.yaml',
                '2021-04-19-172134.yaml', '2021-04-19-171959.yaml', '2021-04-19-171859.yaml', '2021-04-19-171727.yaml', '2021-04-19-171827.yaml']

words =         ['COMPUTER-VISION', 'COMPUTER-VISION', 'COMPUTER-VISION', 'COMPUTER-VISION', 'COMPUTER-VISION',
                'COMPUTER-VISION', 'COMPUTER-VISION', 'COMPUTER-VISION', 'COMPUTER-VISION', 'COMPUTER-VISION']
'''

'''
latentfiles =   ['2021-04-19-172156.yaml', '2021-04-19-171555.yaml', '2021-04-19-171546.yaml', '2021-04-19-173316.yaml', '2021-04-19-172128.yaml',
                '2021-04-19-172134.yaml', '2021-04-19-171959.yaml', '2021-04-19-171859.yaml', '2021-04-19-171727.yaml', '2021-04-19-171827.yaml',
                '2021-04-19-172410.yaml', '2021-04-19-172156.yaml']

words =         ['COMPUTER-VISION-', 'COMPUTER-SCIENCE', 'COMPUTER-VISION-', 'COMPUTER-SCIENCE', 'COMPUTER-VISION-',
                'COMPUTER-SCIENCE', 'COMPUTER-VISION-', 'COMPUTER-SCIENCE', 'COMPUTER-VISION-', 'COMPUTER-SCIENCE',
                'COMPUTER-VISION-', 'COMPUTER-SCIENCE']
'''

'''
latentfiles =   ['2021-04-19-172156.yaml', '2021-04-19-171555.yaml', '2021-04-19-171546.yaml', '2021-04-19-173316.yaml', '2021-04-19-172128.yaml',
                '2021-04-19-172134.yaml', '2021-04-19-171959.yaml', '2021-04-19-171859.yaml', '2021-04-19-171727.yaml', '2021-04-19-171827.yaml',
                '2021-04-19-172410.yaml', '2021-04-19-172156.yaml']

words =         ['TYPE', 'TYPE', 'TYPE', 'TYPE', 'TYPE',
                '256x', '256x', '256x', '256x', '256x',
                'TYPE', 'TYPE']
'''

latentfiles =   ['2021-04-19-173257.yaml'] * (len(ascii_uppercase) + 1)

words =         [c for c in ascii_uppercase]
words.append('A')

latents = []
for file in latentfiles:
    with open(latentdir + file) as f:
        latents.extend(yaml.load(f, Loader=yaml.FullLoader))

model = tf.saved_model.load(modeldir)
latent_dim = 50

chars = [c for c in ascii_uppercase + ascii_lowercase] # 26 + 26
chars += '1234567890.,(!?)+-*/=' # + 21 = 73

def get_random_latent(sd=0.95):
    latent =  np.random.normal(0, sd, size=latent_dim).astype(np.float32)
    return np.tile(latent, (len(text), 1))

def get_random_style(sd=0.2):
    style = np.random.normal(0, sd, size=6)
    style[0] = 0.3
    return style

def get_input(text="default", latent=get_random_latent(), style=None):
    input_latent = latent

    input_labels = []
    for char in text:
        label = [0.0] * len(chars)
        label[chars.index(char)] = 1.0
        if style:
            label.extend(style)
        input_labels.append(label)
    input_labels = np.asarray(input_labels, dtype=np.float32)

    return [input_latent, input_labels]

def save_as_image(output, name="default.png"):
    img_size = output.shape[1]

    imgs = []
    for i in range(output.shape[0]):
        imgs.append(tf.keras.preprocessing.image.array_to_img(output[i]))

    output_img = Image.new('L', (img_size*len(words[0]), img_size))
    for i in range(len(imgs)):
        output_img.paste(imgs[i], (img_size * i, 0))

    output_img.save(save_dir + name)

def get_interpolated_latents(latents, steps=25):
    interpolated = []

    for i in range(len(latents)-1):
        interpolated.extend(np.linspace(latents[i], latents[i+1], num=steps, dtype=np.float32))

    return interpolated

def get_interpolated_labels(words=['LOVE','HATE'], steps=25, style=False):
    input_labels = []
    for word in words:
        char_labels = []
        for char in word:
            label = [0.0] * len(chars)
            label[chars.index(char)] = 1.0
            if style == True:
                label.extend(get_random_style())
            char_labels.append(label)
        input_labels.append(char_labels)
    input_labels = np.asarray(input_labels)

    interpolated = []
    for i in range(len(words)-1):
        interpolated.extend(np.linspace(input_labels[i], input_labels[i+1], num=steps, dtype=np.float32))
    return np.asarray(interpolated)

def get_interpolated_random_latents(num=2, steps=25):
    interpolated = []

    random_latent_01 = get_random_latent()

    for _ in range(num-1):
        random_latent_02 = get_random_latent()
        interpolated.extend(np.linspace(random_latent_01, random_latent_02, num=steps, axis=0))
        random_latent_01 = random_latent_02

    return interpolated

def get_interpolated_styles(styles, steps=25):
    interpolated = []

    style_01 = styles[0]

    for style in styles[1:]:
        interpolated.extend(np.linspace(style_01, style, num=steps, axis=0))
        style_01 = style

    return interpolated

def get_interpolated_random_styles(num=2, steps=25):
    interpolated = []

    style_01 = get_random_style()

    for _ in range(num-1):
        style_02 = get_random_style()
        interpolated.extend(np.linspace(style_01, style_02, num=steps, axis=0))
        style_01 = style_02

    return interpolated

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

latents = get_interpolated_latents(latents=latents, steps=steps)
labels = get_interpolated_labels(words=words, steps=steps)

for i,  [latent, label] in enumerate(zip(latents, labels)):
    latent = [latent] * len(words[0])
    output = model([latent, label])
    save_as_image(output, f'{i:03d}_{text}.png')
    print(f'Saved image {i} of {len(latents)}')

'''
words= ['Liquid', 'Liquid', 'L1qu1d', 'L1QU1D', 'LIQU1D', 'LIQUID', 'LIQUID', 'LIQUID', 'LIQUID', 'LIQUID']

latents = get_interpolated_random_latents(num=len(words), steps=steps)
labels = get_interpolated_labels(words=words ,steps=steps)

for i, [latent, label] in enumerate(zip(latents, labels)):
    output = model([latent, label])
    save_as_image(output, f'{i:03d}_{text}.png')
    print(f'Saved image {i} of {len(labels)}')
'''

'''
latents = get_interpolated_random_latents(20, 50)
styles = get_interpolated_random_styles(20, 50)

for i, [latent, style] in enumerate(zip(latents,styles)):
    input = get_input(text, latent=latent, style=style)
    output = model(input)
    save_as_image(output, f'{i:03d}_{text}.png')
    print(f'Saved image {i} of {len(latents)}')
'''

'''
latents = get_interpolated_random_latents(num=30, steps=15)

for i, latent in enumerate(latents):
    input = get_input(text, latent=latent)
    output = model(input)
    save_as_image(output, f'{i:03d}_{text}.png')
    print(f'Saved image {i} of {len(latents)}')
'''


'''
styles =   [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, -0.5, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.8, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -0.8, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.8, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, -0.8, 0.0, 0.0]]

latent = get_random_latent(sd=0.85)
styles = get_interpolated_styles(styles=styles, steps=25)

for i, style in enumerate(styles):
    input = get_input(text, latent=latent, style=style)
    output = model(input)
    save_as_image(output, f'{i:03d}_{text}.png')
    print(f'Saved image {i} of {len(styles)}')
'''