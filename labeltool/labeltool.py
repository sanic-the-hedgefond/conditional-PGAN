from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QListWidget, QLabel, QSlider
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

from PIL import ImageFont, ImageDraw, Image, ImageQt
from glob import glob
import os

font_dir= 'C:/Users/Schnee/Datasets/Fonts01CleanUp/'
fonts = glob(font_dir + '*.*', recursive=True)

app = QApplication([])
window = QWidget()
layout = QVBoxLayout()
label = QLabel()
font_list = QListWidget()
slider_1 = QSlider(Qt.Horizontal)
slider_2 = QSlider(Qt.Horizontal)

layout.addWidget(label)
layout.addWidget(font_list)
layout.addWidget(slider_1)
layout.addWidget(slider_2)

def select_font(font):
    set_sample_text(font_dir + font.text())

def set_sample_text(font, sample_text="The quick brown fox jumps over the lazy dog", im_size=(1000,100), txt_size=35):
    font = ImageFont.truetype(font, txt_size)
    im = Image.new('L', (im_size[0], im_size[1]), 0)
    draw = ImageDraw.Draw(im)
    w, h = draw.textsize(sample_text, font=font)
    draw.text((im_size[0]/2-w/2, im_size[1]/2-h/2), sample_text, font=font, fill='#FFF')

    qImg = ImageQt.ImageQt(im)
    qPix = QPixmap.fromImage(qImg)
    label.setPixmap(qPix)

for font in fonts:
    filename = os.path.basename(font)
    font_list.addItem(filename)

font_list.itemActivated.connect(select_font)

set_sample_text(fonts[0])

window.setLayout(layout)
window.show()
app.exec()