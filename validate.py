# (C) Samuel Dressel 2020

# Validate trained models with input overlayed by random noises
from tqdm import tqdm

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


PREMODELNAME = "rnn_full_40mfcc_nopreemph_nonoise_resnet_ws08_512"

PATH_TRAINDATA = "/home/smu/Desktop/RNN/validation_data/" + "rnn_full_40mfcc_nopreemph_mixednoise2000_resnet_ws08_512" + "/"
PATH_MODELS = "/home/smu/Desktop/RNN/models/"
PATH_VALIDATE = "/home/smu/Desktop/RNN/validate_mixed/"

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

f = open(PATH_VALIDATE + "40MFCC_NoNoise" + ".txt", "w")
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
