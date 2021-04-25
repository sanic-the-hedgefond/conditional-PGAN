from PIL import ImageFont, ImageDraw, Image
from string import ascii_uppercase, ascii_lowercase
from glob import glob
import os
import numpy as np
import itertools
import yaml
import random

class DatasetGenerator:
    def __init__(self, im_size=4, num_chars=1, step=1, batch_size=16, font_dir='fonts/', num_fonts=0, get_style_labels=True, one_hot=True):
        self.im_size = im_size
        self.num_chars = num_chars
        self.step = step
        self.batch_size = batch_size
        self.font_dir = font_dir
        self.num_fonts = num_fonts
        self.get_style_labels = get_style_labels
        self.one_hot = one_hot

        self.set_fonts()
        self.set_chars()

        self.set_dataset()
        self.set_batch()

    def set_fonts(self):
        self.fonts = []
        font_types = ['*.otf', '*.ttf']
        for type in font_types:
            self.fonts.extend(glob(self.font_dir + type))

        if self.num_fonts != 0:
            self.fonts = self.fonts[:self.num_fonts]
        self.fonts = self.fonts[::self.step]
    
    def randomize_fonts(self):
        random.shuffle(self.fonts)

    def get_num_fonts(self):
        return len(self.fonts)

    def set_chars(self):
        chars = [c for c in ascii_uppercase + ascii_lowercase] # 26 + 26
        chars += '1234567890.,(!?)+-*/=' # + 21 = 73
        self.chars = chars[:self.num_chars]

    def set_dataset(self):
        self.dataset = self.dataset_generator()

    def set_batch(self):
        self.batch = self.batch_generator(self.dataset)

    def reset_generator(self):
        self.set_dataset()
        self.set_batch()

    def dataset_generator(self):
        style_labels = 0
        if self.get_style_labels:
            with open(self.font_dir + '00_labels.yaml') as f:
                style_labels = yaml.load(f, Loader=yaml.FullLoader)

        for font_file in self.fonts:
            font = ImageFont.truetype(font_file, int(self.im_size*(3.0/4.0)))

            random.shuffle(self.chars)
            
            for i, char in enumerate(self.chars):
                im = Image.new('L', (self.im_size, self.im_size), 0)
                draw = ImageDraw.Draw(im)
                w, h = draw.textsize(char, font=font)
                draw.text(((self.im_size-w)/2, (self.im_size-h)/2), char, font=font, fill='#FFF')

                im = np.array(im)/255.0*2.0 - 1.0
                im = np.expand_dims(im, axis=-1)
                
                #create one-hot annotation vector for character classes
                label = np.zeros((len(self.chars)))
                label[i] = 1

                if not self.one_hot:
                    label = i

                #create vector for type characteristics
                if self.get_style_labels:
                    label = np.append(label, style_labels[os.path.basename(font_file)][:-1])

                yield (im, label)

    def batch_generator(self, iterable):
        iterable = iter(iterable)

        while True:
            batch = list(itertools.islice(iterable, self.batch_size * self.num_chars))
            if len(batch) > 0:
                yield batch
            else:
                break

    def save_dataset_as_images(self, output_dir='dataset_output/'):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for font_file in self.fonts:
            font = ImageFont.truetype(font_file, int(self.im_size*(3.0/4.0)))

            for i, char in enumerate(self.chars):
                im = Image.new('L', (self.im_size, self.im_size), 0)
                draw = ImageDraw.Draw(im)
                w, h = draw.textsize(char, font=font)
                draw.text(((self.im_size-w)/2, (self.im_size-h)/2), char, font=font, fill='#FFF')
                
                if(i < 26):
                    im.save(output_dir + '{}_uppercase__{}__{}.png'.format(char, font_file.split('\\')[-1][:-4], self.im_size))
                else:
                    im.save(output_dir + '{}_lowercase__{}__{}.png'.format(char, font_file.split('\\')[-1][:-4], self.im_size))


if __name__ == '__main__':
    dg = DatasetGenerator(64, 1, font_dir='C:/Users/Schnee/Datasets/FreeFonts/')
    dg.save_dataset_as_images('C:/Users/Schnee/Datasets/FreeFontsIMG/')
