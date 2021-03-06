# (C) Samuel Dressel 2020
# Validate trained models with input overlayed by random noises

# Imports for feature extraction
from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav
from tqdm import tqdm
import chromafeatures
import frequencyfeatures
import frequencyandchromafeatures
from pyAudioAnalysis import audioBasicIO

import glob, os, shutil, sys, random
import time
import os.path
from os import path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import utils
from tensorflow.keras import layers
from sklearn.metrics import classification_report
from collections import defaultdict, namedtuple
from typing import List

# Preemph-Filter to reduce noise
PREEMPH = 0.0

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

# Setting up GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

# Paths
PATH_SOURCE = "/home/smu/Desktop/RNN/audiodata/ValidationSET_CAER/"
PATH_TRAINDATA = "/home/smu/Desktop/RNN/audiodata/cache/"
PATH_MODELS = "/home/smu/Desktop/RNN/models/"
PATH_VALIDATE = "/home/smu/Desktop/RNN/validate_caer/"

# Name of the model (for saving and logs)
PREMODELNAME = "rnn_full_mfcc+chroma+time+spec_nopreemph_mixednoise_resnet_ws08_512"

os.chdir(PATH_SOURCE)

print("Generating features from validationsamples ...")

for aud in tqdm(glob.glob("*.wav")):
    [Fs, x] = audioBasicIO.read_audio_file(aud)
    F, f_names = frequencyandchromafeatures.feature_extraction(x, Fs, WINDOW_SIZE*Fs, WINDOW_STEP*Fs)
    (rate,sig) = wav.read(aud)
    mfcc_feat = mfcc(sig, rate, numcep=NUMCEP, nfilt=NUMFILT, winlen=WINDOW_SIZE, winstep=WINDOW_STEP, nfft=NFFT, preemph=PREEMPH)
    emotion = "N"
    if "W" in aud:
        emotion = "W"
    elif "L" in aud:
        emotion = "L"
    elif "E" in aud:
        emotion = "E"
    elif "A" in aud:
        emotion = "A"
    elif "F" in aud:
        emotion = "F"
    elif "T" in aud:
        emotion = "T"
    F = np.swapaxes(F, 0, 1)
    F = np.append(F, mfcc_feat, axis=1)
    featurefile = "../cache/" + aud + "_" + emotion
    np.save(featurefile, F)

names = "0 = Wut, 1 = Langeweile, 2 = Ekel, 3 = Angst, 4 = Freude, 5 = Trauer, 6 = Neutral"

data_test_data = []
ltt = []

print("Generating tensors for validation ...")

os.chdir(PATH_TRAINDATA)

for npyfile in tqdm(glob.glob("*.npy")):
    temp = np.load(npyfile)
    data_test_data.append(temp)

    if "W" in npyfile:
        ltt.append(0)
    elif "L" in npyfile:
        ltt.append(1)
    elif "E" in npyfile:
        ltt.append(2)
    elif "A" in npyfile:
        ltt.append(3)
    elif "F" in npyfile:
        ltt.append(4)
    elif "T" in npyfile:
        ltt.append(5)
    elif "N" in npyfile:
        ltt.append(6)

features_test = tf.convert_to_tensor(data_test_data)
ltt = tf.convert_to_tensor(ltt)
ltt = utils.to_categorical(ltt)
print(ltt.shape)

model = tf.keras.models.load_model(PATH_MODELS + PREMODELNAME + "/")
model.summary()

print("Evaluate validation samples ... ")

results = model.evaluate(features_test, ltt, verbose=0)

predictions = model(features_test, training=False)
pred = np.argmax(model.predict(features_test), axis=-1)
real = np.argmax(ltt, axis=1)

o_classes = classification_report(real, pred)

o_valacc = round((results[1]*100), 2)
o_parameters = PREMODELNAME.split('_')

o_language = o_parameters[1]
if o_language == "full":
    o_language = "German + English"
elif o_language == "ger":
    o_language = "German"
elif o_language == "eng":
    o_language = "English"
else:
    o_language = "-"

o_features = o_parameters[2]
if o_features == "mfcc":
    o_features = "MFCC (13 Features)"
elif o_features == "40mfcc":
    o_features = "MFCC (40 Features)"
elif o_features == "time+spec":
    o_features = "Time and Spectraldomain-Features (8 Features)"
elif o_features == "mfcc+time+spec":
    o_features = "MFCC + Time- and Spectraldomain (48 Features)"
elif o_features == "mfcc+chroma+time+spec":
    o_features = "MFCC + Time- and Spectraldomain + Chroma-Features (61 Features)"
elif o_features == "mfcc+chroma":
    o_features = "MFCC + Chroma-Features (53 Features)"
else:
    o_features = "-"

o_preemph = o_parameters[3]
if o_preemph == "nopreemph":
    o_preemph = "False"
elif o_preemph == "preemph":
    o_preemph = "True"
else:
    o_features = "-"

o_noise = o_parameters[4]
if o_noise == "nonoise":
    o_noise = "False"
elif o_noise == "envnoise":
    o_noise = "True (Enviromental Noise)"
elif o_noise == "mixednoise":
    o_noise = "Mixed (4000 samples)"
elif o_noise == "mixednoise2000":
    o_noise = "Mixed (2000 samples)"
else:
    o_noise = "-"

o_layers = "9xLSTM-RESNET"

o_nfft = "65536"
o_windowsize = "0.8"
o_windowstep = "0.1"

o_language = "Language of the dataset: " + o_language + "\n"
o_features = "Used Features: " + o_features + "\n"
o_preemph = "Preemph-Filter: " + o_preemph + "\n"
o_noise = "Noise: " + o_noise + "\n"
o_layers = "Model: " + o_layers + "\n"
o_nfft = "NFFT-Size: " + o_nfft + "\n"
o_windowsize = "Window-Size: " + o_windowsize + "\n"
o_windowstep = "Window-Step: " + o_windowstep + "\n"
o_valacc = "Validation Accuracy: " + str(o_valacc) + " %\n"

f = open("/home/smu/Desktop/RNN/validate_caer/" + "MFCC+Chroma+TS_MixedNoise4000" + ".txt", "w")
f.write(names)
f.write("\n\n")
f.write(o_classes)
f.write("\n")
f.write(o_language)
f.write(o_features)
f.write(o_preemph)
f.write(o_noise)
f.write(o_layers)
f.write(o_nfft)
f.write(o_windowsize)
f.write(o_windowstep)
f.write(o_valacc)
f.close()
