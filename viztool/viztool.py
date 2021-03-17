from PyQt5.QtWidgets import QApplication, QHBoxLayout, QWidget, QVBoxLayout, QGridLayout, QLabel, QSlider, QPushButton, QFileDialog
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

import numpy as np
import tensorflow as tf
from PIL import Image, ImageQt
import sys
from functools import partial
import os
from datetime import datetime

class Viztool(QWidget):

    def __init__(self):
        super().__init__()

        self.modeldir = 'training/2021-03-12-145730/pgan_stage_2_stabilize/'
        self.num_img = 26
        self.num_chars = 26
        self.latent_dim = 50

        self.slider_steps = 50
        self.slider_per_row = 25

        self.model = tf.saved_model.load(self.modeldir)

        self.input_latent = np.zeros(shape=(1, self.latent_dim), dtype=np.float32)
        self.input_labels = np.zeros((self.num_img, self.num_chars))
        for i in range(self.num_img):
            self.input_labels[i][i % self.num_chars] = 1
        self.input_labels = tf.convert_to_tensor(self.input_labels, dtype=tf.float32)

        self.init_UI()

        self.update_output(self.input_latent, self.input_labels)

    
    def init_UI(self):
        self.layout = QVBoxLayout()

        self.layout_output = QGridLayout()
        self.layout_latent_slider = QGridLayout()

        self.layout.addLayout(self.layout_output)
        self.layout.addLayout(self.layout_latent_slider)

        self.output_label = QLabel()
        self.layout_output.addWidget(self.output_label)

        self.latent_slider_captions = [QLabel() for _ in range(self.latent_dim)]
        self.latent_slider = [QSlider(Qt.Vertical) for _ in range(self.latent_dim)]

        for i in range(self.latent_dim):
            self.latent_slider[i].setTickPosition(QSlider.TickPosition.TicksLeft)
            self.latent_slider[i].setMinimum(-self.slider_steps)
            self.latent_slider[i].setMaximum(self.slider_steps)
            self.latent_slider[i].setMinimumHeight(200)
            self.latent_slider[i].valueChanged.connect(partial(self.slider_changed, i))
            self.latent_slider_captions[i].setText(f'<b>{i+1}</b>')
            self.layout_latent_slider.addWidget(self.latent_slider[i], 0 + 2 * (i // self.slider_per_row), i % self.slider_per_row)
            self.layout_latent_slider.addWidget(self.latent_slider_captions[i], 1 + 2 * (i // self.slider_per_row), i % self.slider_per_row)

        self.layout_btns = QHBoxLayout()

        self.btn_random = QPushButton('Random')
        self.btn_random.clicked.connect(self.randomize_latent)
        self.layout_btns.addWidget(self.btn_random)

        self.btn_select_model = QPushButton('Select Model')
        self.btn_select_model.clicked.connect(self.select_model)
        self.layout_btns.addWidget(self.btn_select_model)

        self.btn_save = QPushButton('Save as Image')
        self.btn_save.clicked.connect(self.save)
        self.layout_btns.addWidget(self.btn_save)

        self.layout.addLayout(self.layout_btns)

        self.setWindowTitle('Viztool')
        self.setLayout(self.layout)
        self.show()

    def save(self):
        savedir = f'{self.modeldir}../viztool/'
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        filename = f'{datetime.now().strftime("%Y-%m-%d-%H%M%S")}.png'
        self.output_img.save(f'{savedir}{filename}')

    def select_model(self):
        modeldir = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        if modeldir:
            model = tf.saved_model.load(self.modeldir)
            if model:
                self.model = model
                self.modeldir = modeldir
            else:
                print("Wrong directory")

    def randomize_latent(self):
        self.input_latent = np.random.normal(0, 0.5, size=(1, self.latent_dim))

        for i in range(self.latent_dim):
            self.latent_slider[i].setValue(self.input_latent[0][i] * self.slider_steps)

        self.update_output(self.input_latent, self.input_labels)

    def slider_changed(self, i):
        self.input_latent[0][i] = self.latent_slider[i].value() / self.slider_steps
        self.update_output(self.input_latent, self.input_labels)

    def update_output(self, input_latent, input_labels, img_width=1600):
        input_latent = tf.convert_to_tensor(input_latent, dtype=tf.float32)
        input_latent = tf.repeat(input_latent, self.num_img, axis=0)

        self.output = self.model([input_latent, input_labels])

        self.img_size = self.output.shape[1]

        imgs = []
        for i in range(self.output.shape[0]):
            imgs.append(tf.keras.preprocessing.image.array_to_img(self.output[i]))

        num_rows = 2
        self.output_img = Image.new('L', (self.num_img//num_rows*self.img_size, self.img_size*num_rows))
        for i in range(len(imgs)):
            self.output_img.paste(imgs[i], (self.img_size*(i%(self.num_img//num_rows)), (i // (self.num_img//num_rows)) * self.img_size))

        img_height = int(self.output_img.size[1] * img_width / self.output_img.size[0])

        self.output_img = self.output_img.resize((img_width, img_height), resample=Image.NEAREST)

        q_img = ImageQt.ImageQt(self.output_img)
        q_pix = QPixmap.fromImage(q_img)
        self.output_label.setPixmap(q_pix)

def main():
    app = QApplication([])
    viztool = Viztool()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()