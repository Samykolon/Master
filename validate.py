import time
import os
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
PATH_VALIDATIONDATA = "/home/smu/Desktop/RNN/validation_data/"
PATH_LOGS = "/home/smu/Desktop/RNN/logs/"
PATH_MODELS = "/home/smu/Desktop/RNN/models/"
PATH_VALIDATE = "/home/smu/Desktop/RNN/validate/"

names = "0 = Wut, 1 = Langeweile, 2 = Ekel, 3 = Angst, 4 = Freude, 5 = Trauer, 6 = Neutral"

subfolders = [ f.name for f in os.scandir(PATH_MODELS) if f.is_dir() ]

for folder in list(subfolders):

    try:
        os.mkdir("/home/smu/Desktop/RNN/validate/" + folder)
    except Exception as e:
        print('Failed to create model folder: %s' % (e))

    data_test_data = []
    ltt = []

    for filename in os.listdir(PATH_VALIDATIONDATA + folder):
        npyfile = PATH_VALIDATIONDATA + folder + "/" + filename
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

    model = tf.keras.models.load_model(PATH_MODELS + folder)
    model.summary()

    results = model.evaluate(features_test, ltt, verbose=0)

    predictions = model(features_test, training=False)

    pred = np.argmax(model.predict(features_test), axis=-1)

    real = np.argmax(ltt, axis=1)

    o_classes = classification_report(real, pred)

    o_valacc = round((results[1]*100), 2)
    o_parameters = folder.split('_')
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
    elif o_features == "timespectral":
        o_features = "Frequency-Domain (8 Features)"
    elif o_features == "21features":
        o_features = "MFCC + Frequency-Domain (21 Features)"
    elif o_features == "34features":
        o_features = "MFCC + Frequency-Domain + Chroma (34 Features)"
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
        o_noise = "Mixed"
    else:
        o_noise = "-"
    o_layers = o_parameters[5]
    if o_layers == "lstm":
        o_layers = "1 x LSTM"
    elif o_layers == "3lstm":
        o_layers = "3 x LSTM"
    else:
        o_layers = "-"
    o_nfft = o_parameters[6]
    o_windowsize = "-"
    o_windowstep = "-"
    if o_nfft == "nfft65536":
        o_nfft = "65536"
        o_windowsize = "0.8"
        o_windowstep = "0.1"
    elif o_nfft == "nfft32768":
        o_nfft = "32768"
        o_windowsize = "0.4"
        o_windowstep = "0.05"
    elif o_nfft == "nfft16384":
        o_nfft = "16384"
        o_windowsize = "0.2"
        o_windowstep = "0.025"
    elif o_nfft == "nfft8192":
        o_nfft = "8192"
        o_windowsize = "0.1"
        o_windowstep = "0.0125"

    o_language = "Language of the dataset: " + o_language + "\n"
    o_features = "Used Features: " + o_features + "\n"
    o_preemph = "Preemph-Filter: " + o_preemph + "\n"
    o_noise = "Noise: " + o_noise + "\n"
    o_layers = "Model: " + o_layers + "\n"
    o_nfft = "NFFT-Size: " + o_nfft + "\n"
    o_windowsize = "Window-Size: " + o_windowsize + "\n"
    o_windowstep = "Window-Step: " + o_windowstep + "\n"
    o_valacc = "Validation Accuracy: " + str(o_valacc) + " %\n"

    f = open("/home/smu/Desktop/RNN/validate/" + folder + "/result.txt", "w")
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
