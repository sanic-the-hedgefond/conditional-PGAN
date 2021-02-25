from PIL import ImageFont, ImageDraw, Image
from string import ascii_uppercase, ascii_lowercase
from glob import glob
import os
import numpy as np
import re

def get_labeled_data(IM_SIZE=4, FONT_DIR='C:/Users/Schnee/Google Drive/Informatik Studium/Semester10/Master/font_GAN/fonts/', OUTPUT=False, num_chars=2):

    fonts = glob(FONT_DIR + '*.*', recursive=True)
    chars = []

    for char in ascii_uppercase:
        chars.append(char)

    if num_chars == 52:
        for char in ascii_lowercase:
            chars.append(char)

    elif num_chars == 1:
        chars = ['A']

    elif num_chars == 2:
        chars = ['A', 'O']

    OUTPUT_DIR = 'fonts/dataset/' + str(IM_SIZE) + '/'
    if OUTPUT and not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    print('Found {} fonts each with {} characters selected.'.format(len(fonts), len(chars)))
    print('Characters: {}'.format(chars))
    
    images = [[]] * len(chars)
    labels = [[]] * len(chars)

    #dataset = []

    for font_file in fonts:
        font = ImageFont.truetype(font_file, int(IM_SIZE*(3.0/4.0)))

        #font_images = [[]] * len(chars)
        #font_labels = [[]] * len(chars)

        for i, char in enumerate(chars):
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
            #im = np.array(im)

            label = i

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

            #font_images.append(im)
            #font_labels.append(label)

            images[i].append(im)
            labels[i].append(label)

        #dataset.append([np.asarray(font_images), np.asarray(font_labels)])

        #images.append(font_images)
        #labels.append(font_labels)

        #images.append(np.asarray(font_images))
        #labels.append(np.asarray(font_labels))

    #return np.asarray(images), np.asarray(labels)
    #return dataset
    return [np.asarray(images), np.asarray(labels)]

test = get_labeled_data()
print(test[0].shape)
print(test[1].shape)