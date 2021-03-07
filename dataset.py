from PIL import ImageFont, ImageDraw, Image
from string import ascii_uppercase, ascii_lowercase
from glob import glob
import os
import numpy as np
import itertools
#import re

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
        self.fonts =  fonts[::self.step]

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

                '''
                #create one-hot annotation vector + 3 for font characteristics
                label = np.zeros((len(chars) + 3))
                label[i] = 1

                if re.search('thin', font_file, re.IGNORECASE):
                    label[-3] = 0
                elif re.search(('light'), font_file, re.IGNORECASE):
                    label[-3] = 0.2
                elif re.search(('bold'), font_file, re.IGNORECASE):
                    label[-3] = 0.6
                elif re.search(('heavy'), font_file, re.IGNORECASE):
                    label[-3] = 0.8
                elif re.search(('black'), font_file, re.IGNORECASE):
                    label[-3] = 1
                else:
                    label[-3] = 0.4

                # identify width
                if re.search('cond', font_file, re.IGNORECASE):
                    label[-2] = 0
                elif re.search('extended', font_file, re.IGNORECASE):
                    label[-2] = 1
                else:
                    label[-2] = 0.5

                # identify italic
                if re.search('italic', font_file, re.IGNORECASE) or re.search('oblique', font_file, re.IGNORECASE):
                    label[-1] = 1
                else:
                    label[-1] = 0
                '''

                label = i

                yield (im, np.asarray(label))

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
            im = Image.new('L', (self.im_size, self.im_size), 0)
            draw = ImageDraw.Draw(im)
            w, h = draw.textsize(char, font=font)
            draw.text(((self.im_size-w)/2, (self.im_size-h)/2), char, font=font, fill='#FFF')

            for i, char in enumerate(self.chars):
                    if(i < 26):
                        im.save(output_dir + '{}_uppercase__{}__{}.png'.format(char, font_file.split('\\')[-1][:-4], self.im_size))
                    else:
                        im.save(output_dir + '{}_lowercase__{}__{}.png'.format(char, font_file.split('\\')[-1][:-4], self.im_size))