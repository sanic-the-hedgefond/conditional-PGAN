from PyQt5.QtWidgets import QApplication, QHBoxLayout, QWidget, QVBoxLayout, QGridLayout, QLabel, QSlider, QPushButton, QFileDialog
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

import numpy as np
import tensorflow as tf

from PIL import Image, ImageQt

import os
import sys
from functools import partial
from datetime import datetime
import yaml

class Viztool(QWidget):

    def __init__(self):
        super().__init__()

        #self.modeldir = 'training/2021-03-19-102857/models/pcgan_stage_5_stabilize/'
        #self.modeldir = 'training/2021-04-17-122742/models/pcgan_stage_1_stabilize/'
        #self.modeldir = 'C:/Users/Schnee/Desktop/MASTER/Training_Processes/pcgan/2021-04-01-115036/models/pcgan_stage_4_stabilize/'
        #self.modeldir = 'C:/Users/Schnee/Desktop/MASTER/Training_Processes/pcgan/2021-04-03-201328/models/pcgan_stage_5_stabilize_1/'
        #self.modeldir = 'C:/Users/Schnee/Desktop/MASTER/Training_Processes/pcgan/2021-04-06-123539/models/pcgan_stage_4_stabilize_1/'
        #self.modeldir = 'C:/Users/Schnee/Desktop/MASTER/Training_Processes/pcgan/2021-04-06-175938/models/pcgan_stage_4_stabilize_2/'
        #self.modeldir = 'C:/Users/Schnee/Desktop/MASTER/Training_Processes/pcgan/2021-04-08-141135/models/pcgan_stage_4_stabilize_8/'
        #self.modeldir = 'C:/Users/Schnee/Desktop/MASTER/Training_Processes/pcgan/2021-04-08-141135/models/pcgan_stage_5_stabilize_3/'
        #self.modeldir = 'C:/Users/Schnee/Desktop/MASTER/Training_Processes/pcgan/2021-04-10-192347/models/pcgan_stage_6_fade_in_2/'
        #self.modeldir = 'C:/Users/Schnee/Desktop/MASTER/Training_Processes/pcgan/2021-04-13-113046/models/pcgan_stage_4_fade_in/'
        #self.modeldir = 'C:/Users/Schnee/Desktop/MASTER/Training_Processes/pcgan/2021-04-13-113046/models/pcgan_tmp/'
        #self.modeldir = 'C:/Users/Schnee/Desktop/MASTER/Training_Processes/pcgan/2021-04-14-073102/models/pcgan_stage_5_fade_in/'
        #self.modeldir = 'C:/Users/Schnee/Desktop/MASTER/Training_Processes/pcgan/2021-04-14-073102/models/pcgan_tmp/'
        #self.modeldir = 'C:/Users/Schnee/Desktop/MASTER/Training_Processes/pcgan/2021-04-17-193005/models/pcgan_stage_4_stabilize/'
        #self.modeldir = 'C:/Users/Schnee/Desktop/MASTER/Training_Processes/pcgan/2021-04-17-232232/models/pcgan_tmp/'
        self.modeldir = 'C:/Users/Schnee/Desktop/MASTER/Training_Processes/pcgan/2021-04-18-070339/models/pcgan_stage_4_fade_in/'
        
        
        self.num_img = 73 #26 #52 #73
        self.num_chars = 73 #26 #52 #73
        self.latent_dim = 50 #20
        self.random_sd = 0.0
        self.img_per_batch = 12
        self.num_rows_chars = 4 #2

        self.label_names = ['Weight', 'Width', 'Contrast', 'Serifs', 'Italic'] #, 'Roundness']

        self.slider_steps = 50
        self.slider_per_row = 25
        

        self.model = tf.saved_model.load(self.modeldir)

        self.update_output_flag = True

        self.input_latent = np.zeros(shape=(1, self.latent_dim), dtype=np.float32)
        self.input_labels = np.zeros((self.num_img, self.num_chars + len(self.label_names)))
        for i in range(self.num_img):
            self.input_labels[i][i % self.num_chars] = 1
        
        self.input_style = np.zeros(len(self.label_names))

        self.init_UI()

        self.randomize_latent()

        self.update_output(self.input_latent, self.input_labels)

    
    def init_UI(self):
        self.layout = QVBoxLayout()

        self.layout_output = QGridLayout()
        self.layout_random_slider = QHBoxLayout()
        self.layout_latent_slider = QGridLayout()
        self.layout_style_slider = QGridLayout()

        self.random_slider = QSlider(Qt.Horizontal)
        self.random_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.random_slider.setMinimum(0)
        self.random_slider.setMaximum(20)
        self.random_slider.valueChanged.connect(self.random_slider_changed)

        self.random_slider_caption = QLabel()
        self.random_slider_caption.setText('Latent Vector Standard Deviation: 0.0')

        self.layout_random_slider.addWidget(self.random_slider_caption)
        self.layout_random_slider.addWidget(self.random_slider)

        self.layout.addLayout(self.layout_output)
        self.layout.addLayout(self.layout_random_slider)
        self.layout.addLayout(self.layout_latent_slider)
        self.layout.addLayout(self.layout_style_slider)

        self.output_label = QLabel()
        self.layout_output.addWidget(self.output_label)

        self.latent_slider_captions = [QLabel() for _ in range(self.latent_dim)]
        self.latent_slider = [QSlider(Qt.Vertical) for _ in range(self.latent_dim)]

        for i in range(self.latent_dim):
            self.latent_slider[i].setTickPosition(QSlider.TickPosition.TicksLeft)
            self.latent_slider[i].setMinimum(-self.slider_steps*2)
            self.latent_slider[i].setMaximum(self.slider_steps*2)
            self.latent_slider[i].setMinimumHeight(200)
            self.latent_slider[i].valueChanged.connect(partial(self.slider_changed, i))
            self.latent_slider_captions[i].setText(f'<b>{i+1}</b>')
            self.layout_latent_slider.addWidget(self.latent_slider[i], 0 + 2 * (i // self.slider_per_row), i % self.slider_per_row)
            self.layout_latent_slider.addWidget(self.latent_slider_captions[i], 1 + 2 * (i // self.slider_per_row), i % self.slider_per_row)

        self.style_slider_captions = [QLabel() for _ in range(len(self.label_names))]
        self.style_slider = [QSlider(Qt.Horizontal) for _ in range(len(self.label_names))]

        for i in range(len(self.label_names)):
            self.style_slider[i].setTickPosition(QSlider.TickPosition.TicksLeft)
            self.style_slider[i].setMinimum(-self.slider_steps)
            self.style_slider[i].setMaximum(self.slider_steps)
            self.style_slider[i].valueChanged.connect(partial(self.style_slider_changed, i))
            self.style_slider_captions[i].setText(f'<b>{self.label_names[i]}</b>')
            self.layout_style_slider.addWidget(self.style_slider_captions[i], i, 0)
            self.layout_style_slider.addWidget(self.style_slider[i], i, 1)

        self.layout_btns = QHBoxLayout()
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
        with open(f'{savedir}{filename[:-4]}.yaml', 'w') as file:
            yaml.dump(self.input_latent.tolist(), file)

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
        self.input_latent = np.random.normal(0, self.random_sd, size=(1, self.latent_dim))
        self.update_output_flag = False

        for i in range(self.latent_dim):
            self.latent_slider[i].setValue(self.input_latent[0][i] * self.slider_steps)

        self.update_output_flag = True
        self.update_output(self.input_latent, self.input_labels)

    def slider_changed(self, i):
        self.input_latent[0][i] = self.latent_slider[i].value() / self.slider_steps
        if self.update_output_flag:
            self.update_output(self.input_latent, self.input_labels)

    def random_slider_changed(self):
        self.random_sd = self.random_slider.value() / 10
        self.random_slider_caption.setText(f'Latent Vector Standard Deviation: {self.random_slider.value() / 10}')

    def style_slider_changed(self, i):
        self.input_labels[:,self.num_chars + i] = self.style_slider[i].value() / self.slider_steps
        self.update_output(self.input_latent, self.input_labels)

    def update_output(self, input_latent, input_labels, img_width=1600):
        input_latent = tf.convert_to_tensor(input_latent, dtype=tf.float32)
        input_latent = tf.repeat(input_latent, self.num_img, axis=0)

        input_labels = tf.convert_to_tensor(input_labels, dtype=tf.float32)

        #img_per_batch = self.img_per_batch
        self.output = []
        for i in range(self.num_img // self.img_per_batch):
            index_start = i * self.img_per_batch
            index_end = min((i+1) * self.img_per_batch, self.num_img)
            self.output.extend(self.model([input_latent[index_start:index_end], input_labels[index_start:index_end]]))

        self.output = (np.asarray(self.output) * 0.5) + 0.5
        self.output = 1 - self.output
        self.img_size = self.output.shape[1]

        imgs = []
        for i in range(self.output.shape[0]):
            imgs.append(tf.keras.preprocessing.image.array_to_img(self.output[i]))

        self.output_img = Image.new('L', (self.num_img//self.num_rows_chars*self.img_size, self.img_size*self.num_rows_chars))
        for i in range(len(imgs)):
            self.output_img.paste(imgs[i], (self.img_size*(i%(self.num_img//self.num_rows_chars)), (i // (self.num_img//self.num_rows_chars)) * self.img_size))

        img_height = int(self.output_img.size[1] * img_width / self.output_img.size[0])

        self.output_img = self.output_img.resize((img_width, img_height), resample=Image.NEAREST)

        q_pix = QPixmap.fromImage(ImageQt.ImageQt(self.output_img))
        self.output_label.setPixmap(q_pix)

def main():
    app = QApplication([])
    viztool = Viztool()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()