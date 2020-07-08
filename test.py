import pyaudio
import wave
import time
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav


model = tf.keras.models.load_model('models/rnn_1')
model.summary()

# Input-Device
INPUT_DEVICE = "AT2020 USB: Audio (hw:2,0)" # Name of the input device

# Lookup the index of the desired Input-Device, make sure jack is running
pa = pyaudio.PyAudio()
chosen_device_index = -1
for x in range(0,pa.get_device_count()):
    info = pa.get_device_info_by_index(x)
    # print (pa.get_device_info_by_index(x))
    if info["name"] == INPUT_DEVICE:
        chosen_device_index = info["index"]

print ("Chosen index: ", chosen_device_index)
pa.terminate()

plt.ion()

class DynamicUpdate():
    # X RANGE
    min_x = 0.0
    max_x = 1.0
    # Samplerate of the input
    SAMPLERATE = 48000
    # NFFT - This is the frequency resolution
    # By default, the FFT size is the first equal or superior power of 2 of the window size.
    # If we have a samplerate of 16000 Hz and a window size of 32 ms, we get 512 samples in each window.
    # The next superior power would be 512 so we choose that
    NFFT = 512
    # Format to read in audio data
    FORMAT = pyaudio.paInt16
    # Size of the Window
    WINDOW_SIZE = 0.032
    # Window step Size = Window-Duration/8 - Overlapping Parameter
    WINDOW_STEP = 0.004
    # Preemph-Filter to reduce noise
    PREEMPH = 0.97
    # Record Seconds
    RECORD_SECONDS = 20

    def __init__(self, model, device):
        self.varw = 0.0
        self.varl = 0.0
        self.vare = 0.0
        self.vara = 0.0
        self.varf = 0.0
        self.vart = 0.0
        self.varn = 0.0
        self.model = model
        self.inputdevice = device
        # Chunk - Each second we calculate 1/0.004 featuresets, so to get the chunk it has to be samplerate/(1/Windowstep)
        self.CHUNK = int(self.SAMPLERATE/(1/self.WINDOW_STEP))

    def on_launch(self, xdata, ydata):
        self.figure, self.ax = plt.subplots()
        self.figure.canvas.set_window_title('Emotion Recognition')
        self.ax.set_ylim(self.min_x, self.max_x)

    def on_running(self, xdata, ydata):
        self.ax.cla()
        self.ax.bar(xdata, ydata)
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()


    def __call__(self):
        xdata = ['Wut','Langeweile','Ekel','Angst','Freude','Trauer','Neutral']
        ydata = [self.varw,self.varl,self.vare,self.vara,self.varf,self.vart,self.varn]
        self.on_launch(xdata, ydata)
        frames = []
        result = []
        p = pyaudio.PyAudio()
        stream = p.open(format=self.FORMAT, channels=1, rate=self.SAMPLERATE, input_device_index=self.inputdevice, input=True, frames_per_buffer=self.CHUNK)
        for i in range(0, int(self.SAMPLERATE / self.CHUNK * self.RECORD_SECONDS)):
            data = stream.read(self.CHUNK, exception_on_overflow = False)
            decoded = np.frombuffer(data, 'int16')
            mfcc_feat = mfcc(decoded, samplerate=self.SAMPLERATE/3, winlen=self.WINDOW_SIZE, winstep=self.WINDOW_STEP, nfft=self.NFFT)
            if len(frames) < 299:
                frames.append(mfcc_feat)
            elif len(frames) >= 299:
                predict_test = tf.convert_to_tensor(frames)
                predict_test = tf.transpose(predict_test, [1, 0, 2])
                result = model.predict(predict_test)
                ydata = [result.item(0), result.item(1), result.item(2), result.item(3), result.item(4), result.item(5), result.item(6)]
                self.on_running(xdata, ydata)
                frames = []

        stream.stop_stream()
        stream.close()
        p.terminate()

d = DynamicUpdate(model, chosen_device_index)
d()


# import matplotlib.pyplot as plt
# from pandas import DataFrame
# from itertools import cycle, islice
# import numpy as np
# import random
# import time
#
# p = pyaudio.PyAudio()
# stream = p.open(format=FORMAT, channels=1, rate=SAMPLERATE, input_device_index=chosen_device_index, input=True, frames_per_buffer=CHUNK)
# frames = []
# result = []
#
# for i in range(0, int(SAMPLERATE / CHUNK * RECORD_SECONDS)):
#     data = stream.read(CHUNK, exception_on_overflow = False)
#     decoded = np.frombuffer(data, 'int16')
#     mfcc_feat = mfcc(decoded, samplerate=SAMPLERATE/3, winlen=WINDOW_SIZE, winstep=WINDOW_STEP, nfft=NFFT)
#     if len(frames) < 299:
#         frames.append(mfcc_feat)
#     elif len(frames) >= 299:
#         predict_test = tf.convert_to_tensor(frames)
#         predict_test = tf.transpose(predict_test, [1, 0, 2])
#         result = model.predict(predict_test)
#         frames = []
#         frames.append(mfcc_feat)
#         print(result)
#
#
# stream.stop_stream()
# stream.close()
# p.terminate()
# print(result)
#
#
#
# plt.ion()
#
# class DynamicUpdate():
#     #Suppose we know the x range
#     min_x = 0.0
#     max_x = 1.0
#
#     def __init__(self, w, l, e, a, f, t, n):
#         self.varw = w
#         self.varl = l
#         self.vare = e
#         self.vara = a
#         self.varf = f
#         self.vart = t
#         self.varn = n
#
#     def on_launch(self, xdata, ydata):
#         #Set up plot
#         self.figure, self.ax = plt.subplots()
#         self.figure.canvas.set_window_title('Emotion Recognition')
#         self.ax.set_ylim(self.min_x, self.max_x)
#
#     def on_running(self, xdata, ydata):
#         self.ax.bar(xdata, ydata)
#         self.figure.canvas.draw()
#         self.figure.canvas.flush_events()
#
#
#     def __call__(self):
#         xdata = ['Wut','Langeweile','Ekel','Angst','Freude','Trauer','Neutral']
#         ydata = [self.varw,self.varl,self.vare,self.vara,self.varf,self.vart,self.varn]
#         self.on_launch(xdata, ydata)
#         self.on_running(xdata, ydata)
#
#
#         return xdata, ydata
#
# # d = DynamicUpdate(result.item(0), result.item(1), result.item(2), result.item(3), result.item(4), result.item(5), result.item(6))
# d = DynamicUpdate(rr.item(0, 0), rr.item(0, 1), rr.item(0, 2), rr.item(0, 3), rr.item(0, 4), rr.item(0, 5), rr.item(0, 6))
# d()
#
# import operator
# rw = rr.item(0, 0)
# rl = rr.item(0, 1)
# re = rr.item(0, 2)
# ra = rr.item(0, 3)
# rf = rr.item(0, 4)
# rt = rr.item(0, 5)
# rn = rr.item(0, 6)
#
# rlist = []
# rlist.append(rw)
# rlist.append(rl)
# rlist.append(re)
# rlist.append(ra)
# rlist.append(rf)
# rlist.append(rt)
# rlist.append(rn)
#
# max_value = max(rlist)
# max_index = rlist.index(max_value)
#
# if max_index == 0:
#     print("Wut")
# elif max_index == 1:
#     print("Langeweile")
# elif max_index == 2:
#     print("Ekel")
# elif max_index == 3:
#     print("Angst")
# elif max_index == 4:
#     print("Freude")
# elif max_index == 5:
#     print("Trauer")
# elif max_index == 6:
#     print("Neutral")
