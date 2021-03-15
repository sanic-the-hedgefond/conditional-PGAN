from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QListWidget, QLabel, QSlider, QPushButton
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

from PIL import ImageFont, ImageDraw, Image, ImageQt
from glob import glob
import os
import yaml

font_dir = 'C:/Users/Schnee/Datasets/Fonts01CleanUp/'
font_types = ['*.otf', '*.ttf']
fonts = []
for type in font_types:
    fonts.extend(glob(font_dir + type))

label_file = 'labels.yaml'
current_row = 0
current_font = fonts[0]

slider_steps = 100

labels = dict()

if not os.path.exists(font_dir + label_file):
    for font in fonts:
        filename = os.path.basename(font)
        labels[filename] = [0.0, 0.0]

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

slider_1 = QSlider(Qt.Horizontal)

layout.addWidget(caption)
layout.addWidget(font_view)
layout.addWidget(font_list)

layout.addWidget(btn_next)
layout.addWidget(btn_prev)
layout.addWidget(btn_save)

layout.addWidget(slider_1)

def next_font():
    global current_row
    current_row = (current_row + 1) % (len(fonts))
    next_font = font_list.item(current_row)
    set_sample_text(font_dir + next_font.text())
    #caption.setText(f'{len(fonts)} fonts. Current font: {next_font.text()}')

def prev_font():
    global current_row
    current_row = (current_row - 1) % (len(fonts))
    next_font = font_list.item(current_row)
    set_sample_text(font_dir + next_font.text())
    #caption.setText(f'{len(fonts)} fonts. Current font: {next_font.text()}')

def select_font(font):
    global current_row
    current_row = font_list.currentRow()
    set_sample_text(font_dir + font.text())
    #caption.setText(f'{len(fonts)} fonts. Current font: {font.text()}')

def set_sample_text(font, sample_text="The quick brown fox jumps over the lazy dog", im_size=(1000,100), txt_size=35):
    global current_font
    current_font = os.path.basename(font)
    #caption.setText(f'{current_row+1}/{len(fonts)} fonts. Current font: {current_font}. Labels: {labels[current_font]}')
    update_caption()
    font = ImageFont.truetype(font, txt_size)
    im = Image.new('L', (im_size[0], im_size[1]), 0)
    draw = ImageDraw.Draw(im)
    w, h = draw.textsize(sample_text, font=font)
    draw.text((im_size[0]/2-w/2, im_size[1]/2-h/2), sample_text, font=font, fill='#FFF')

    q_img = ImageQt.ImageQt(im)
    q_pix = QPixmap.fromImage(q_img)
    font_view.setPixmap(q_pix)

    slider_1.setValue(labels[current_font][0])

def slider1_changed():
    labels[current_font][0] = slider_1.value()
    update_caption()

def update_caption():
    caption.setText(f'{current_row+1}/{len(fonts)} fonts. Current font: {current_font}. Labels: {labels[current_font][0] / slider_steps}')

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

slider_1.setMinimum(-100)
slider_1.setMaximum(100)
slider_1.valueChanged.connect(slider1_changed)

select_font(font_list.item(current_row))

window.setLayout(layout)
window.show()
app.exec()