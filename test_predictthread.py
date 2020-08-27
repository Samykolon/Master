import pyaudio
import wave
import time
import os
import os.path
from os import path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import utils
from tensorflow.keras import layers
from tensorflow.python.client import device_lib
import matplotlib.pyplot as plt
from python_speech_features import mfcc
from python_speech_features import logfbank
from pyAudioAnalysis import audioBasicIO
import scipy.io.wavfile as wav

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

model = tf.keras.models.load_model('models/rnn_ger_40mfcc_nopreemph_nonoise_resnet_ws08_512')
model.summary()

plt.ion()

class DynamicUpdate():
    # X RANGE
    min_x = 0.0
    max_x = 1.0

    # Samplerate of the input
    SAMPLERATE = 48000
    # NFFT - This is the frequency resolution
    # By default, the FFT size is the first equal or superior power of 2 of the window size.
    # If we have a samplerate of 48000 Hz and a window size of 800 ms, we get 38400 samples in each window.
    # The next superior power would be 65536 so we choose that
    NFFT = 65536

    # Size of the Window
    WINDOW_SIZE = 0.8

    # Window step Size = Window-Duration/8 - Overlapping Parameter
    WINDOW_STEP = 0.1

    # Units for Training
    UNITS = 512

    # Number of MFCCs
    NUMCEP = 40

    # Number of Melfilters
    NUMFILT = 40

    # Preemph-Filter to reduce noise
    PREEMPH = 0.0

    # Record Seconds
    RECORD_SECONDS = 6

    WAVE_INPUT_FILENAME = "/home/smu/Desktop/RNN/TEST.wav"

    def __init__(self, model):
        self.varw = 0.0
        self.varl = 0.0
        self.vare = 0.0
        self.vara = 0.0
        self.varf = 0.0
        self.vart = 0.0
        self.varn = 0.0
        self.model = model

    def on_launch(self, xdata, ydata):
        self.figure, self.ax = plt.subplots()
        self.figure.canvas.set_window_title('Emotion Recognition')
        self.ax.set_ylim(self.min_x, self.max_x)

    def on_running(self, xdata):
        if path.exists(self.WAVE_INPUT_FILENAME):
            self.ax.cla()
            (rate,sig) = wav.read(self.WAVE_INPUT_FILENAME)
            mfcc_feat = mfcc(sig, rate, numcep=self.NUMCEP, nfilt=self.NUMFILT, winlen=self.WINDOW_SIZE, winstep=self.WINDOW_STEP, nfft=self.NFFT, preemph=self.PREEMPH)
            predict_test = tf.convert_to_tensor(mfcc_feat)
            predict_test = tf.expand_dims(predict_test, 0)
            result = model.predict(predict_test)
            ydata = [result.item(0), result.item(1), result.item(2), result.item(3), result.item(4), result.item(5), result.item(6)]
            self.ax.bar(xdata, ydata)
            self.figure.canvas.draw()
            self.figure.canvas.flush_events()


    def __call__(self):
        xdata = ['Wut','Langeweile','Ekel','Angst','Freude','Trauer','Neutral']
        ydata = [self.varw,self.varl,self.vare,self.vara,self.varf,self.vart,self.varn]
        self.on_launch(xdata, ydata)
        starttime = time.time()
        while True:
            self.on_running(xdata)
            time.sleep(6.0 - ((time.time() - starttime) % 6.0))



d = DynamicUpdate(model)
d()
