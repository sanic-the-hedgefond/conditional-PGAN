from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QListWidget, QLabel, QSlider, QPushButton, QShortcut
from PyQt5.QtGui import QPixmap, QKeySequence
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

slider_steps = 20

label_names = ['Weight', 'Width', 'Contrast', 'Serifs', 'Italic', 'Roundness']
labels = dict()

label_cache = [0.0] * len(label_names)

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

layout = QHBoxLayout()

layout_left = QVBoxLayout()
layout_right = QVBoxLayout()

layout.addLayout(layout_left)
layout.addLayout(layout_right)

font_list = QListWidget()
caption = QLabel()
font_view = QLabel()

btn_prev = QPushButton('Prev (Left Arrow Key)')
btn_next = QPushButton('Next (Right Arrow Key)')
btn_copy = QPushButton('Copy Labels (Strg+C)')
btn_paste = QPushButton('Paste Labels (Strg+V))')
btn_save = QPushButton('Save (Return)')

slider_captions = [QLabel() for _ in range(len(label_names))]
sliders = [QSlider(Qt.Horizontal) for _ in range(len(label_names))]

layout_left.addWidget(font_list)

layout_right.addWidget(caption)
layout_right.addWidget(font_view)

layout_sliders = QGridLayout()

for i in range(len(label_names)):
    slider_captions[i].setText(label_names[i])
    layout_sliders.addWidget(slider_captions[i], i, 0)
    layout_sliders.addWidget(sliders[i], i, 1)

layout_right.addLayout(layout_sliders)

layout_btns = QHBoxLayout()

layout_btns.addWidget(btn_prev)
layout_btns.addWidget(btn_next)
layout_btns.addWidget(btn_copy)
layout_btns.addWidget(btn_paste)
layout_btns.addWidget(btn_save)

layout_right.addLayout(layout_btns)

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

def set_sample_text(font, sample_text="ABCDEFGHIJKLMNOPQRSTUVWXYZ\nabcdefghijklmnopqrstuvwxyz .,-!?/", im_size=(1200,200), txt_size=55):
    global current_font
    current_font = os.path.basename(font)
    update_caption()
    font = ImageFont.truetype(font, txt_size)
    im = Image.new('L', (im_size[0], im_size[1]), 0)
    draw = ImageDraw.Draw(im)
    _, h = draw.textsize(sample_text, font=font)
    draw.text((20, im_size[1]/2-h/2), sample_text, font=font, fill='#FFF')

    q_img = ImageQt.ImageQt(im)
    q_pix = QPixmap.fromImage(q_img)
    font_view.setPixmap(q_pix)

    for i, slider in enumerate(sliders):
        slider.setValue(labels[current_font][i] * slider_steps)

def slider_changed(i):
    labels[current_font][i] = sliders[i].value() / slider_steps
    update_caption()

def update_caption():
    caption.setText(f'Font {current_row+1} of {len(fonts)}. Current font: {current_font}. Labels: {labels[current_font]}')

def save_labels():
    with open(font_dir + label_file, 'w') as f:
        yaml.dump(labels, f)

def copy_label():
    global label_cache
    label_cache = labels[current_font]

def paste_label():
    global labels
    global label_cache
    labels[current_font] = label_cache

    for i, slider in enumerate(sliders):
        slider.setValue(labels[current_font][i] * slider_steps)

for font in fonts:
    filename = os.path.basename(font)
    font_list.addItem(filename)

font_list.itemActivated.connect(select_font)

btn_next.clicked.connect(next_font)
btn_prev.clicked.connect(prev_font)
btn_copy.clicked.connect(copy_label)
btn_paste.clicked.connect(paste_label)
btn_save.clicked.connect(save_labels)

btn_next.setShortcut(QKeySequence('Right'))
btn_prev.setShortcut(QKeySequence('Left'))
btn_copy.setShortcut(QKeySequence('Ctrl+C'))
btn_paste.setShortcut(QKeySequence('Ctrl+V'))
btn_save.setShortcut(QKeySequence('Return'))

for i in range(len(sliders)):
    sliders[i].setMinimum(-slider_steps)
    sliders[i].setMaximum(slider_steps)
    sliders[i].valueChanged.connect(partial(slider_changed, i))

select_font(font_list.item(current_row))

window.setLayout(layout)
window.show()
app.exec()