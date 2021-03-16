from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QListWidget, QLabel, QSlider, QPushButton
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

from PIL import ImageFont, ImageDraw, Image, ImageQt
from glob import glob
import os
import yaml
from functools import partial

font_dir = 'C:/Users/Schnee/Datasets/Fonts01CleanUp/'
font_types = ['*.otf', '*.ttf']

fonts = []
for type in font_types:
    fonts.extend(glob(font_dir + type))

label_file = '00_labels.yaml'
current_row = 0
current_font = os.path.basename(fonts[current_row])

slider_steps = 100

label_names = ['Weight', 'Width', 'Contrast', 'Serifs', 'Italic', 'Roundness']
labels = dict()

if not os.path.exists(font_dir + label_file):
    for font in fonts:
        filename = os.path.basename(font)
        labels[filename] = [0.0] * len(label_names)

    with open(font_dir + label_file, 'w') as f:
        yaml.dump(labels, f)

else:
    with open(font_dir + label_file) as f:
        labels = yaml.load(f, Loader=yaml.FullLoader)

app = QApplication([])
window = QWidget()
layout = QVBoxLayout()

caption = QLabel()
font_view = QLabel()
font_list = QListWidget()

btn_next = QPushButton('Next')
btn_prev = QPushButton('Prev')
btn_save = QPushButton('Save')

slider_captions = [QLabel() for _ in range(len(label_names))]
sliders = [QSlider(Qt.Horizontal) for _ in range(len(label_names))]

layout.addWidget(caption)
layout.addWidget(font_view)
layout.addWidget(font_list)

layout.addWidget(btn_next)
layout.addWidget(btn_prev)
layout.addWidget(btn_save)

for i in range(len(label_names)):
    slider_captions[i].setText(label_names[i])
    layout.addWidget(slider_captions[i])
    layout.addWidget(sliders[i])

def next_font():
    global current_row
    current_row = (current_row + 1) % (len(fonts))
    font = font_list.item(current_row)
    set_sample_text(font_dir + font.text())

def prev_font():
    global current_row
    current_row = (current_row - 1) % (len(fonts))
    font = font_list.item(current_row)
    set_sample_text(font_dir + font.text())

def select_font(font):
    global current_row
    current_row = font_list.currentRow()
    set_sample_text(font_dir + font.text())

def set_sample_text(font, sample_text="The quick brown fox jumps over the lazy dog", im_size=(1000,100), txt_size=35):
    global current_font
    current_font = os.path.basename(font)
    update_caption()
    font = ImageFont.truetype(font, txt_size)
    im = Image.new('L', (im_size[0], im_size[1]), 0)
    draw = ImageDraw.Draw(im)
    w, h = draw.textsize(sample_text, font=font)
    draw.text((im_size[0]/2-w/2, im_size[1]/2-h/2), sample_text, font=font, fill='#FFF')

    q_img = ImageQt.ImageQt(im)
    q_pix = QPixmap.fromImage(q_img)
    font_view.setPixmap(q_pix)

    for i, slider in enumerate(sliders):
        slider.setValue(labels[current_font][i] * slider_steps)

def slider_changed(i):
    labels[current_font][i] = sliders[i].value() / slider_steps
    update_caption()

def update_caption():
    caption.setText(f'{current_row+1}/{len(fonts)} fonts. Current font: {current_font}. Labels: {labels[current_font]}')

def save_labels():
    with open(font_dir + label_file, 'w') as f:
        yaml.dump(labels, f)

for font in fonts:
    filename = os.path.basename(font)
    font_list.addItem(filename)

font_list.itemActivated.connect(select_font)
btn_next.clicked.connect(next_font)
btn_prev.clicked.connect(prev_font)
btn_save.clicked.connect(save_labels)

for i in range(len(sliders)):
    sliders[i].setMinimum(-slider_steps)
    sliders[i].setMaximum(slider_steps)
    sliders[i].valueChanged.connect(partial(slider_changed, i))

select_font(font_list.item(current_row))

window.setLayout(layout)
window.show()
app.exec()