from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav
from tqdm import tqdm

import glob, os, shutil, sys, random
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import utils
from tensorflow.keras import layers

# Preemph-Filter to reduce noise
PREEMPH = 0.97

# Number of Testsamples
NUMBER_TESTSAMPLES = 230   # Number of Testsamples

# Name of the model (for saving and logs)
MODELNAME = "rnn_full_mfcc_nopreemph_nonoise_3lstm"

# NFFT - This is the frequency resolution
# By default, the FFT size is the first equal or superior power of 2 of the window size.
# If we have a samplerate of 16000 Hz and a window size of 32 ms, we get 512 samples in each window.
# The next superior power would be 512 so we choose that
NFFT = 2048

# Size of the Window
WINDOW_SIZE = 0.032

# Window step Size = Window-Duration/8 - Overlapping Parameter
WINDOW_STEP = 0.004

# Path where the train-data is stored
PATH_TRAINDATA = "/home/smu/Desktop/RNN/train_data/"
#Path where the test-data is stored - gets randomly picked out of traindata
PATH_TESTDATA = "/home/smu/Desktop/RNN/test_data/"

os.chdir("/home/smu/Desktop/RNN/audiodata/own_sixseconds")

print("Generating features from own recordings ...")

for aud in tqdm(glob.glob("*.wav")):
    (rate,sig) = wav.read(aud)
    mfcc_feat = mfcc(sig, rate, winlen=WINDOW_SIZE, winstep=WINDOW_STEP, nfft=NFFT, preemph=PREEMPH)
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

os.chdir("/home/smu/Desktop/RNN/audiodata/emo_sixseconds")

print("Generating features from emoDB ...")

for aud in tqdm(glob.glob("*.wav")):
    (rate,sig) = wav.read(aud)
    mfcc_feat = mfcc(sig, rate, winlen=WINDOW_SIZE, winstep=WINDOW_STEP, nfft=NFFT, preemph=PREEMPH)
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
    featurefile = "../../train_data/" + aud + "_" + emotion
    np.save(featurefile, mfcc_feat)

os.chdir("/home/smu/Desktop/RNN/audiodata/zenodo_sixseconds")

print("Generating features from zenodo-database...")

for aud in tqdm(glob.glob("*.wav")):
    (rate,sig) = wav.read(aud)
    mfcc_feat = mfcc(sig, rate, winlen=WINDOW_SIZE, winstep=WINDOW_STEP, nfft=NFFT, preemph=PREEMPH)
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

print("Chosing test samples ...")

for filename in os.listdir(PATH_TESTDATA):
    file_path = os.path.join(PATH_TESTDATA, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

for x in range(NUMBER_TESTSAMPLES):
    random_file=random.choice(os.listdir(PATH_TRAINDATA))
    src = PATH_TRAINDATA + random_file
    dst = PATH_TESTDATA + random_file
    shutil.move(src,dst)

print("Initialising GPU ...")

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

data_train_data = []
ltr = []
data_test_data = []
ltt = []

os.chdir(PATH_TRAINDATA)

print("Generating tensors for training ...")

for txtfile in tqdm(glob.glob("*.npy")):

    temp = np.load(txtfile)
    data_train_data.append(temp)

    if "W" in txtfile:
        ltr.append(0)
    elif "L" in txtfile:
        ltr.append(1)
    elif "E" in txtfile:
        ltr.append(2)
    elif "A" in txtfile:
        ltr.append(3)
    elif "F" in txtfile:
        ltr.append(4)
    elif "T" in txtfile:
        ltr.append(5)
    elif "N" in txtfile:
        ltr.append(6)

features_train = tf.convert_to_tensor(data_train_data)
ltr = tf.convert_to_tensor(ltr)
ltr = utils.to_categorical(ltr)

os.chdir(PATH_TESTDATA)

print("Generating tensors for testing ...")

for txtfile in tqdm(glob.glob("*.npy")):

    temp = np.load(txtfile)
    data_test_data.append(temp)

    if "W" in txtfile:
        ltt.append(0)
    elif "L" in txtfile:
        ltt.append(1)
    elif "E" in txtfile:
        ltt.append(2)
    elif "A" in txtfile:
        ltt.append(3)
    elif "F" in txtfile:
        ltt.append(4)
    elif "T" in txtfile:
        ltt.append(5)
    elif "N" in txtfile:
        ltt.append(6)

features_test = tf.convert_to_tensor(data_test_data)
ltt = tf.convert_to_tensor(ltt)
ltt = utils.to_categorical(ltt)

print("Tensors generated, moving testfiles back ...")

files = os.listdir(PATH_TESTDATA)

for f in files:
    shutil.copy2(PATH_TESTDATA + f, PATH_TRAINDATA)

print("Generating model ...")

model = tf.keras.Sequential()
model.add(layers.LSTM((512), input_shape=(1493, 13), return_sequences=True))
model.add(layers.Dropout(0.4))
model.add(layers.LSTM((512), input_shape=(1493, 13), return_sequences=True))
model.add(layers.Dropout(0.4))
model.add(layers.LSTM((512), input_shape=(1493, 13)))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(7, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

print("Training ...")
os.chdir("/home/smu/Desktop/RNN")
log_dir = "logs/" + MODELNAME
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history = model.fit(features_train, ltr, epochs=25, batch_size=128, validation_data=(features_test, ltt), callbacks=[tensorboard_callback])

model_dir = 'models/' + MODELNAME
model.save(model_dir)

print("Model trained and saved!")

# Eine Studie über Emotionserkennung mithilfe menschlicher Stimme mit rekurrenten neuronalen Netzen
# A study on emotion recognition using human voice with Recurrent Neural Networks
