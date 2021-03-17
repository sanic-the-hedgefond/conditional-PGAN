from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QListWidget, QLabel, QSlider, QPushButton, QShortcut
from PyQt5.QtGui import QPixmap, QKeySequence
from PyQt5.QtCore import Qt

import numpy as np
import tensorflow as tf
from PIL import Image, ImageQt
import sys

class Viztool(QWidget):

    def __init__(self):
        super().__init__()

        self.modeldir = 'training/2021-03-12-145730/pgan_stage_2_stabilize/'
        self.num_img = 26
        self.num_chars = 26
        self.latent_dim = 50

        self.model = tf.saved_model.load(self.modeldir)

        self.input_latent = tf.random.normal(shape=[self.num_img, self.latent_dim], dtype=tf.float32)
        self.input_labels = np.zeros((self.num_img, self.num_chars))
        for i in range(self.num_img):
            self.input_labels[i][i % self.num_chars] = 1
        self.input_labels = tf.convert_to_tensor(self.input_labels, dtype=tf.float32)

        self.output = self.model([self.input_latent, self.input_labels])

        self.img_size = self.output.shape[1]

        imgs = []
        for i in range(self.output.shape[0]):
            imgs.append(tf.keras.preprocessing.image.array_to_img(self.output[i]))

        self.output_img = Image.new('L', (self.num_img*self.img_size, self.img_size))
        for i in range(len(imgs)):
            self.output_img.paste(imgs[i], (self.img_size*i, 0))

        self.initUI()

        q_img = ImageQt.ImageQt(self.output_img)
        q_pix = QPixmap.fromImage(q_img)
        self.output_label.setPixmap(q_pix)
    
    def initUI(self):
        self.layout = QVBoxLayout()

        self.layout_output = QGridLayout()
        self.layout_latent_slider = QGridLayout()

        self.layout.addLayout(self.layout_output)
        self.layout.addLayout(self.layout_latent_slider)

        self.output_label = QLabel()

        self.layout_output.addWidget(self.output_label)

        self.setWindowTitle('Viztool')
        self.setLayout(self.layout)
        self.show()

def main():
    app = QApplication([])
    viztool = Viztool()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()