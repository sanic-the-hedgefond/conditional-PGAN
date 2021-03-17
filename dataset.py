from PIL import ImageFont, ImageDraw, Image
from string import ascii_uppercase, ascii_lowercase
from glob import glob
import os
import numpy as np
import itertools

class DatasetGenerator:
    def __init__(self, im_size=4, num_chars=1, step=1, batch_size=16, font_dir='fonts/'):
        self.im_size = im_size
        self.num_chars = num_chars
        self.step = step
        self.batch_size = batch_size
        self.font_dir = font_dir

        self.set_fonts()
        self.set_chars()

        self.set_dataset()
        self.set_batch()

    def set_fonts(self):
        fonts = glob(self.font_dir + '*.*', recursive=True)
        self.fonts = fonts[::self.step]

    def get_num_fonts(self):
        return len(self.fonts)

    def set_chars(self):
        chars = [c for c in ascii_uppercase + ascii_lowercase]
        self.chars = chars[:self.num_chars]

    def set_dataset(self):
        self.dataset = self.dataset_generator()

    def set_batch(self):
        self.batch = self.batch_generator(self.dataset)

    def reset_generator(self):
        self.set_dataset()
        self.set_batch()

    def dataset_generator(self):
        for font_file in self.fonts:
            font = ImageFont.truetype(font_file, int(self.im_size*(3.0/4.0)))

            for i, char in enumerate(self.chars):
                im = Image.new('L', (self.im_size, self.im_size), 0)
                draw = ImageDraw.Draw(im)
                w, h = draw.textsize(char, font=font)
                draw.text(((self.im_size-w)/2, (self.im_size-h)/2), char, font=font, fill='#FFF')

                im = np.array(im)/255.0*2.0 - 1.0
                im = np.expand_dims(im, axis=-1)
                
                #create one-hot annotation vector
                label = np.zeros((len(self.chars)))
                label[i] = 1

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