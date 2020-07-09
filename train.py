from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav

import glob, os, shutil, sys, random
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import utils
from tensorflow.keras import layers

NUMBER_MFCC = 13           # Number of MFCCs
NUMBER_FRAMES = 1493       # Numer of Frames
NUMBER_TESTSAMPLES = 250   # Number of Testsamples
BATCH_SIZE = 5             # Batch Size
CLASSES = 7

# Samplerate of the input
SAMPLERATE = 16000

# NFFT - This is the frequency resolution
# By default, the FFT size is the first equal or superior power of 2 of the window size.
# If we have a samplerate of 16000 Hz and a window size of 32 ms, we get 512 samples in each window.
# The next superior power would be 512 so we choose that
NFFT = 512

# Size of the Window
WINDOW_SIZE = 0.032

# Window step Size = Window-Duration/8 - Overlapping Parameter
WINDOW_STEP = 0.004

# Preemph-Filter to reduce noise
PREEMPH = 0.97

os.chdir("/home/smu/Desktop/RNN/own_sixseconds")

print("Generating features from own recordings...")

for aud in glob.glob("*.wav"):
    (rate,sig) = wav.read(aud)
    mfcc_feat = mfcc(sig, rate, winlen=0.064, winstep=0.008, nfft=4096)

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
    if len(mfcc_feat) == 1493:
        featurefile = "../train_data/" + aud + "_" + emotion
        np.save(featurefile, mfcc_feat)

os.chdir("/home/smu/Desktop/RNN/emo_sixseconds")

print("Generating features from emoDB ...")

for aud in glob.glob("*.wav"):
    (rate,sig) = wav.read(aud)
    mfcc_feat = mfcc(sig, rate, winlen=WINDOW_SIZE, winstep=WINDOW_STEP, nfft=NFFT)
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
    featurefile = "../train_data/" + aud + "_" + emotion
    np.save(featurefile, mfcc_feat)

os.chdir("/home/smu/Desktop/RNN/zenodo_sixseconds")

print("Generating features from zenodo-database...")

for aud in glob.glob("*.wav"):
    (rate,sig) = wav.read(aud)
    mfcc_feat = mfcc(sig, rate, winlen=WINDOW_SIZE, winstep=WINDOW_STEP, nfft=2048)
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
    if len(mfcc_feat) == 1493:
        featurefile = "../train_data/" + aud + "_" + emotion
        np.save(featurefile, mfcc_feat)

path = "/home/smu/Desktop/RNN/train_data/"
moveto = "/home/smu/Desktop/RNN/test_data/"

print("Chosing test samples...")

for filename in os.listdir(moveto):
    file_path = os.path.join(moveto, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

for x in range(NUMBER_TESTSAMPLES):
    random_file=random.choice(os.listdir("/home/smu/Desktop/RNN/train_data"))
    src = path + random_file
    dst = moveto + random_file
    shutil.move(src,dst)

data_train_data = []
label_train_data = []
ltr = []

os.chdir("/home/smu/Desktop/RNN/train_data")

print("Generating tensors for training...")

for txtfile in glob.glob("*.npy"):

    temp = np.load(txtfile)
    data_train_data.append(temp)


    if "W" in txtfile:
        array = np.full(NUMBER_FRAMES, 0)
        label_train_data.append(array)
        ltr.append(0)
    elif "L" in txtfile:
        array = np.full(NUMBER_FRAMES, 1)
        label_train_data.append(array)
        ltr.append(1)
    elif "E" in txtfile:
        array = np.full(NUMBER_FRAMES, 2)
        label_train_data.append(array)
        ltr.append(2)
    elif "A" in txtfile:
        array = np.full(NUMBER_FRAMES, 3)
        label_train_data.append(array)
        ltr.append(3)
    elif "F" in txtfile:
        array = np.full(NUMBER_FRAMES, 4)
        label_train_data.append(array)
        ltr.append(4)
    elif "T" in txtfile:
        array = np.full(NUMBER_FRAMES, 5)
        label_train_data.append(array)
        ltr.append(5)
    elif "N" in txtfile:
        array = np.full(NUMBER_FRAMES, 6)
        label_train_data.append(array)
        ltr.append(6)

features_train = tf.convert_to_tensor(data_train_data)
labels_train = tf.convert_to_tensor(label_train_data)
ltr = tf.convert_to_tensor(ltr)
ltr = utils.to_categorical(ltr)

labeled_train_data = tf.data.Dataset.from_tensors((features_train, ltr))

print(labeled_train_data)

data_test_data = []
label_test_data = []
ltt = []

os.chdir("/home/smu/Desktop/RNN/test_data")

print("Generating tensors for testing...")

for txtfile in glob.glob("*.npy"):


    temp = np.load(txtfile)
    data_test_data.append(temp)


    if "W" in txtfile:
        array = np.full(NUMBER_FRAMES, 0)
        label_test_data.append(array)
        ltt.append(0)
    elif "L" in txtfile:
        array = np.full(NUMBER_FRAMES, 1)
        label_test_data.append(array)
        ltt.append(1)
    elif "E" in txtfile:
        array = np.full(NUMBER_FRAMES, 2)
        label_test_data.append(array)
        ltt.append(2)
    elif "A" in txtfile:
        array = np.full(NUMBER_FRAMES, 3)
        label_test_data.append(array)
        ltt.append(3)
    elif "F" in txtfile:
        array = np.full(NUMBER_FRAMES, 4)
        label_test_data.append(array)
        ltt.append(4)
    elif "T" in txtfile:
        array = np.full(NUMBER_FRAMES, 5)
        label_test_data.append(array)
        ltt.append(5)
    elif "N" in txtfile:
        array = np.full(NUMBER_FRAMES, 6)
        label_test_data.append(array)
        ltt.append(6)

features_test = tf.convert_to_tensor(data_test_data)
labels_test = tf.convert_to_tensor(label_test_data)
ltt = tf.convert_to_tensor(ltt)
ltt = utils.to_categorical(ltt)

labeled_test_data = tf.data.Dataset.from_tensors((features_test, ltt))

print(labeled_test_data)

print("Generating model...")

model = tf.keras.Sequential()
model.add(layers.LSTM((100), input_shape=(1493, 13), return_sequences=True))
model.add(layers.Dropout(0.4))
model.add(layers.LSTM((100), input_shape=(1493, 13), return_sequences=True))
model.add(layers.Dropout(0.2))
model.add(layers.LSTM((100), input_shape=(1493, 13)))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(7, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

print("Training...")

history = model.fit(features_train, ltr, epochs=25, batch_size=64, validation_data=(features_test, ltt))

os.chdir("/home/smu/Desktop/RNN")

model.save('models/rnn_full_3lstm')

print("Model trained and saved!")

# Frequenzbereiche anpassen
# Noise hinzufügen

# Eine Studie über Emotionserkennung mithilfe menschlicher Stimme mit rekurrenten neuronalen Netzen
# A study on emotion recognition using human voice with Recurrent Neural Networks
