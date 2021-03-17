from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QListWidget, QLabel, QSlider, QPushButton, QShortcut
from PyQt5.QtGui import QPixmap, QKeySequence
from PyQt5.QtCore import Qt

from PIL import ImageFont, ImageDraw, Image, ImageQt
from glob import glob
import os
import yaml
from functools import partial
import copy
import sys

class Labeltool(QWidget):

    def __init__(self):
        super().__init__()

        label_file = '00_labels.yaml'
        font_types = ['*.otf', '*.ttf']
        self.font_dir = 'C:/Users/Schnee/Datasets/Fonts01CleanUp/'

        self.fonts = []
        for type in font_types:
            self.fonts.extend(glob(self.font_dir + type))

        self.current_row = 0
        self.current_font = os.path.basename(self.fonts[self.current_row])

        self.slider_steps = 20

        self.label_names = ['Weight', 'Width', 'Contrast', 'Serifs', 'Italic', 'Roundness']
        self.labels = dict()

        self.label_cache = [0.0] * len(self.label_names)

        if not os.path.exists(self.font_dir + label_file):
            for font in self.fonts:
                filename = os.path.basename(font)
                labels[filename] = [0.0] * len(self.label_names)

            with open(self.font_dir + label_file, 'w') as f:
                yaml.dump(self.labels, f)

        else:
            with open(self.font_dir + label_file) as f:
                self.labels = yaml.load(f, Loader=yaml.FullLoader)

        self.initUI()

    def initUI(self):
        self.layout = QHBoxLayout()

        self.layout_left = QVBoxLayout()
        self.layout_right = QVBoxLayout()

        self.layout.addLayout(self.layout_left)
        self.layout.addLayout(self.layout_right)

        self.font_list = QListWidget()
        self.caption = QLabel()
        self.font_view = QLabel()

        self.btn_prev = QPushButton('Prev (Left Arrow Key)')
        self.btn_next = QPushButton('Next (Right Arrow Key)')
        self.btn_copy = QPushButton('Copy Labels (Ctrl+C)')
        self.btn_paste = QPushButton('Paste Labels (Ctrl+V)')
        self.btn_save = QPushButton('Save (Ctrl+S)')

        self.slider_captions = [QLabel() for _ in range(len(self.label_names))]
        self.sliders = [QSlider(Qt.Horizontal) for _ in range(len(self.label_names))]

        self.layout_left.addWidget(self.font_list)

        self.layout_right.addWidget(self.caption)
        self.layout_right.addWidget(self.font_view)

        self.layout_sliders = QGridLayout()

        for i in range(len(self.label_names)):
            self.slider_captions[i].setText(self.label_names[i])
            self.sliders[i].setTickPosition(QSlider.TicksBelow)
            self.sliders[i].setTickInterval(2)
            self.layout_sliders.addWidget(self.slider_captions[i], i, 0)
            self.layout_sliders.addWidget(self.sliders[i], i, 1)

        self.layout_right.addLayout(self.layout_sliders)

        layout_btns = QHBoxLayout()

        layout_btns.addWidget(self.btn_prev)
        layout_btns.addWidget(self.btn_next)
        layout_btns.addWidget(self.btn_copy)
        layout_btns.addWidget(self.btn_paste)
        layout_btns.addWidget(self.btn_save)

        self.layout_right.addLayout(layout_btns)

        for font in self.fonts:
            filename = os.path.basename(font)
            self.font_list.addItem(filename)

        self.font_list.itemActivated.connect(self.select_font)

        self.btn_next.clicked.connect(self.next_font)
        self.btn_prev.clicked.connect(self.prev_font)
        self.btn_copy.clicked.connect(self.copy_label)
        self.btn_paste.clicked.connect(self.paste_label)
        self.btn_save.clicked.connect(self.save_labels)

        self.btn_next.setShortcut(QKeySequence('Right'))
        self.btn_prev.setShortcut(QKeySequence('Left'))
        self.btn_copy.setShortcut(QKeySequence('Ctrl+C'))
        self.btn_paste.setShortcut(QKeySequence('Ctrl+V'))
        self.btn_save.setShortcut(QKeySequence('Ctrl+S'))

        for i in range(len(self.sliders)):
            self.sliders[i].setMinimum(-self.slider_steps)
            self.sliders[i].setMaximum(self.slider_steps)
            self.sliders[i].valueChanged.connect(partial(self.slider_changed, i))

        self.select_font(self.font_list.item(self.current_row))

        self.setWindowTitle('Labeltool')
        self.setLayout(self.layout)
        self.show()

    def next_font(self):
        self.current_row = (self.current_row + 1) % (len(self.fonts))
        font = self.font_list.item(self.current_row)
        self.set_sample_text(self.font_dir + font.text())

    def prev_font(self):
        self.current_row = (self.current_row - 1) % (len(self.fonts))
        font = self.font_list.item(self.current_row)
        self.set_sample_text(self.font_dir + font.text())

    def select_font(self, font):
        self.current_row = self.font_list.currentRow()
        self.set_sample_text(self.font_dir + font.text())

    def set_sample_text(self, font, sample_text="ABCDEFGHIJKLMNOPQRSTUVWXYZ\nabcdefghijklmnopqrstuvwxyz .,-!?/", im_size=(1200,200), txt_size=55):
        self.current_font = os.path.basename(font)
        self.update_caption()
        font = ImageFont.truetype(font, txt_size)
        im = Image.new('L', (im_size[0], im_size[1]), 0)
        draw = ImageDraw.Draw(im)
        _, h = draw.textsize(sample_text, font=font)
        draw.text((20, im_size[1]/2-h/2), sample_text, font=font, fill='#FFF')

        q_img = ImageQt.ImageQt(im)
        q_pix = QPixmap.fromImage(q_img)
        self.font_view.setPixmap(q_pix)

        for i, slider in enumerate(self.sliders):
            slider.setValue(self.labels[self.current_font][i] * self.slider_steps)

    def slider_changed(self, i):
        self.labels[self.current_font][i] = self.sliders[i].value() / self.slider_steps
        self.update_caption()

    def update_caption(self):
        self.caption.setText(f'<b>Font {self.current_row+1} of {len(self.fonts)}. Current font: {self.current_font}. Labels: {self.labels[self.current_font]}</b>')

    def save_labels(self):
        with open(self.font_dir + label_file, 'w') as f:
            yaml.dump(self.labels, f)

    def copy_label(self):
        self.label_cache = copy.copy(self.labels[self.current_font])

    def paste_label(self):
        self.labels[self.current_font] = copy.copy(self.label_cache)

        for i, slider in enumerate(self.sliders):
            slider.setValue(self.labels[self.current_font][i] * self.slider_steps)

def main():
    app = QApplication([])
    labeltool = Labeltool()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()