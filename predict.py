import pyaudio
import wave
import time
import os
import os.path
from os import path
import numpy as np
import matplotlib.pyplot as plt
from python_speech_features import mfcc
from python_speech_features import logfbank
from pyAudioAnalysis import audioBasicIO
import scipy.io.wavfile as wav
import pyaudio
import audioop
import glob, shutil, sys, random
from pydub import AudioSegment
from pydub.utils import make_chunks
import soundfile as sf
import json
import requests
from ctypes import *
from contextlib import contextmanager

class DynamicUpdate():
    # X RANGE
    min_x = 0.0
    max_x = 1.0

    # Samplerate of the input
    SAMPLERATE = 48000

    # Format of the audio (bitdepth)
    FORMAT = pyaudio.paInt16

    # Number of channels
    CHANNELS = 2

    # Chunksize for recording
    CHUNK = 1000

    # File for temporary audio saving
    WAVE_OUTPUT_FILENAME = "/home/smu/Desktop/RNN/TEST.wav"

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

    # Data for Plot
    xdata = ['Wut','Langeweile','Ekel','Angst','Freude','Trauer','Neutral']
    ydata = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Input Device
    INPUT_DEVICE = "AT2020 USB: Audio"

    def paintplot(self, xdata, ydata):
        y_pos = np.arange(len(xdata))
        plt.ylim(0.0,1.0)
        plt.bar(y_pos, ydata)
        plt.xticks(y_pos, xdata)
        plt.tight_layout()
        plt.show()

    def record_and_calculate(self):
        print("Looking for audio input device ... ")
        pa = pyaudio.PyAudio()
        chosen_device_index = -1
        for x in range(0,pa.get_device_count()):
            info = pa.get_device_info_by_index(x)
            if self.INPUT_DEVICE in info["name"]:
                chosen_device_index = info["index"]
        print(chosen_device_index)
        pa.terminate()

        print("Start recording - speak for 6 seconds ...")
        audio = pyaudio.PyAudio()
        stream = audio.open(format=self.FORMAT, channels=self.CHANNELS,
                        rate=self.SAMPLERATE, input=True,
                        frames_per_buffer=self.CHUNK, input_device_index=chosen_device_index)

        frames = []

        for i in range(0, int(self.SAMPLERATE / self.CHUNK * self.RECORD_SECONDS)):
            data = stream.read(self.CHUNK, exception_on_overflow = False)
            frames.append(data)

        print("Done Recording!")
        print("Calculating features ...")

        waveFile = wave.open(self.WAVE_OUTPUT_FILENAME, 'wb')
        waveFile.setnchannels(self.CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(self.FORMAT))
        waveFile.setframerate(self.SAMPLERATE)
        waveFile.writeframes(b''.join(frames))
        waveFile.close()
        sound = AudioSegment.from_file(self.WAVE_OUTPUT_FILENAME)
        sound = sound.set_channels(1)
        change_in_dBFS = -40.0 - sound.dBFS
        sound = sound.apply_gain(change_in_dBFS)
        sound.export(self.WAVE_OUTPUT_FILENAME, format="wav")

        stream.stop_stream()
        stream.close()
        audio.terminate()

        (rate,sig) = wav.read(self.WAVE_OUTPUT_FILENAME)
        mfcc_feat = mfcc(sig, rate, numcep=self.NUMCEP, nfilt=self.NUMFILT, winlen=self.WINDOW_SIZE, winstep=self.WINDOW_STEP, nfft=self.NFFT, preemph=self.PREEMPH)
        mfcc_feat = np.expand_dims(mfcc_feat, axis=0)

        data = json.dumps({"signature_name": "serving_default",
                           "instances": mfcc_feat.tolist()})

        headers = {"content-type": "application/json"}

        json_response = requests.post('http://localhost:9000/v1/models/emotiondetection/versions/1:predict', data=data, headers=headers)

        result = json.loads(json_response.text)

        result = result["predictions"]

        self.ydata = [result[0][0], result[0][1], result[0][2], result[0][3], result[0][4], result[0][5], result[0][6]]

    def __call__(self):
        self.record_and_calculate()
        self.paintplot(self.xdata, self.ydata)



d = DynamicUpdate()
d()
