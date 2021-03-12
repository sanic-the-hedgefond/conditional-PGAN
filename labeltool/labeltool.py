from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout
import pyqtgraph as pg

from PIL import ImageFont, ImageDraw, Image
from glob import glob
import numpy as np

sample_text = "Lorem ipsum dolor"
im_size = (800,100)

font_dir= 'C:/Users/Schnee/Datasets/Fonts01CleanUp/'

fonts = glob(font_dir + '*.*', recursive=True)

font = ImageFont.truetype(fonts[0], 20)
im = Image.new('L', (im_size[0], im_size[1]), 0)
draw = ImageDraw.Draw(im)
w, h = draw.textsize(sample_text, font=font)
draw.text(((im_size[0])/2, (im_size[1])/2), sample_text, font=font, fill='#FFF')

im = np.array(im)/255.0*2.0 - 1.0
im = np.expand_dims(im, axis=-1)

#pg.image(im)


app = QApplication([])
window = QWidget()
layout = QVBoxLayout()
layout.addWidget(QPushButton('Top'))
layout.addWidget(QPushButton('Bottom'))
layout.addWidget(pg.image(im))
window.setLayout(layout)
window.show()
app.exec()