from PIL import ImageFont, ImageDraw, Image
from string import ascii_uppercase, ascii_lowercase
from glob import glob
import os
import numpy as np
import re
import itertools

class DatasetGenerator:
    def __init__(self, im_size=4, num_chars=1, step=1, batch_size=16, font_dir='C:/Users/Schnee/Google Drive/Informatik Studium/Semester10/Master/font_GAN/fonts/'):
        self.im_size = im_size
        self.num_chars = num_chars
        self.step = step
        self.batch_size = batch_size
        self.font_dir = font_dir

        self.set_fonts()
        self.set_chars()

        self.dataset = self.dataset_generator()
        self.batch = self.batch_generator(self.dataset)

    def set_fonts(self):
        fonts = glob(self.font_dir + '*.*', recursive=True)
        self.fonts =  fonts[::self.step]

    def get_num_fonts(self):
        return len(self.fonts)

    def set_chars(self):
        chars = [c for c in ascii_uppercase + ascii_lowercase]
        self.chars = chars[:self.num_chars]

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

    def get_batch(self):
        return next(self.batch)

    def save_dataset_as_images(self, output_dir='dataset_output/'):
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

DG = DatasetGenerator()

batch = DG.get_batch()
print(len(batch))
print(batch[0])

def get_num_fonts(step=1, FONT_DIR='C:/Users/Schnee/Google Drive/Informatik Studium/Semester10/Master/font_GAN/fonts/'):
    fonts = glob(FONT_DIR + '*.*', recursive=True)
    return len(fonts[::step])

def data_generator(IM_SIZE=4, num_chars=2, step=1, FONT_DIR='C:/Users/Schnee/Google Drive/Informatik Studium/Semester10/Master/font_GAN/fonts/', OUTPUT=False, OUTPUT_DIR='C:/Users/Schnee/Desktop/Test/'):

    fonts = glob(FONT_DIR + '*.*', recursive=True)
    fonts = fonts[::step]
    chars = []

    for char in (ascii_uppercase + ascii_lowercase):
        chars.append(char)

    chars = chars[:num_chars]

    OUTPUT_DIR = OUTPUT_DIR + str(IM_SIZE) + '/'
    if OUTPUT and not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    print('Found {} fonts each with {} characters selected.'.format(len(fonts), len(chars)))
    print('Characters: {}'.format(chars))

    for i, font_file in enumerate(fonts):
        #images = np.zeros((num_chars, IM_SIZE, IM_SIZE, 1))
        #labels = np.zeros((num_chars))

        font = ImageFont.truetype(font_file, int(IM_SIZE*(3.0/4.0)))

        for j, char in enumerate(chars):
            im = Image.new('L', (IM_SIZE, IM_SIZE), 0)
            draw = ImageDraw.Draw(im)
            w, h = draw.textsize(char, font=font)
            draw.text(((IM_SIZE-w)/2, (IM_SIZE-h)/2), char, font=font, fill='#FFF')

            if OUTPUT:
                if(i < 26):
                    im.save(OUTPUT_DIR + '{}_uppercase__{}__{}.png'.format(char, font_file.split('\\')[-1][:-4], IM_SIZE))
                else:
                    im.save(OUTPUT_DIR + '{}_lowercase__{}__{}.png'.format(char, font_file.split('\\')[-1][:-4], IM_SIZE))

            # shift range from [0,255] to [-1,1]
            im = np.array(im)/255.0*2.0 - 1.0
            im = np.expand_dims(im, axis=-1)

            label = j

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

            #images[j] = np.asarray(im)
            #labels[j] = label

            yield (im, np.asarray(label))

    #images = np.asarray(images)
    #labels = np.asarray(labels)

    #images = np.reshape(images, (num_chars, len(fonts), IM_SIZE, IM_SIZE, 1))
    #labels = np.reshape(labels, (num_chars, len(fonts)))

    #return [images, labels]

def batch_generator(iterable, batch_size=1, num_chars=1):
    iterable = iter(iterable)

    while True:
        batch = list(itertools.islice(iterable, batch_size * num_chars))
        if len(batch) > 0:
            yield batch
        else:
            break

#dg = data_generator(num_chars=2)

#bg = batch_generator(dg, 4, 2)
#print(next(bg)[::2])

#print(dg)


#print(next(batches[0]))