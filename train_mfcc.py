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
PREEMPH = 0.0

# Number of Testsamples
NUMBER_TESTSAMPLES = 200  # Number of Testsamples

# Name of the model (for saving and logs)
MODELNAME = "rnn_full_mfcc_nopreemph_nonoise_lstm_nfft65536"

# NFFT - This is the frequency resolution
# By default, the FFT size is the first equal or superior power of 2 of the window size.
# If we have a samplerate of 48000 Hz and a window size of 32 ms, we get 1536 samples in each window.
# The next superior power would be 2048 so we choose that
NFFT = 65536

# Size of the Window
WINDOW_SIZE = 0.8

# Window step Size = Window-Duration/8 - Overlapping Parameter
WINDOW_STEP = 0.1

# Path where the train-data is stored
PATH_TRAINDATA = "/home/smu/Desktop/RNN/train_data/"
# Path where the test-data is stored - gets randomly picked out of traindata
PATH_TESTDATA = "/home/smu/Desktop/RNN/test_data/"
# Path for the validation_data for later testing
PATH_VALIDATIONDATA = "/home/smu/Desktop/RNN/validation_data/"
# Path for the temporal saved weights
PATH_WEIGHTS = "/home/smu/Desktop/RNN/temp/"

# os.chdir("/home/smu/Desktop/RNN/audiodata/own_sixseconds")
#
# print("Generating features from own recordings ...")
#
# for aud in tqdm(glob.glob("*.wav")):
#     (rate,sig) = wav.read(aud)
#     mfcc_feat = mfcc(sig, rate, winlen=WINDOW_SIZE, winstep=WINDOW_STEP, nfft=NFFT, preemph=PREEMPH)
#     emotion = "N"
#     if "W" in aud:
#         emotion = "W"
#     elif "L" in aud:
#         emotion = "L"
#     elif "E" in aud:
#         emotion = "E"
#     elif "A" in aud:
#         emotion = "A"
#     elif "F" in aud:
#         emotion = "F"
#     elif "T" in aud:
#         emotion = "T"
#     featurefile = "../../train_data/" + aud + "_" + emotion
#     np.save(featurefile, mfcc_feat)
#
# # os.chdir("/home/smu/Desktop/RNN/audiodata/own_sixseconds_envnoise")
# #
# # print("Generating features from own recordings with noise ...")
# #
# # for aud in tqdm(glob.glob("*.wav")):
# #     (rate,sig) = wav.read(aud)
# #     mfcc_feat = mfcc(sig, rate, winlen=WINDOW_SIZE, winstep=WINDOW_STEP, nfft=NFFT, preemph=PREEMPH)
# #     emotion = "N"
# #     if "W" in aud:
# #         emotion = "W"
# #     elif "L" in aud:
# #         emotion = "L"
# #     elif "E" in aud:
# #         emotion = "E"
# #     elif "A" in aud:
# #         emotion = "A"
# #     elif "F" in aud:
# #         emotion = "F"
# #     elif "T" in aud:
# #         emotion = "T"
# #     featurefile = "../../train_data/" + aud + "___" + emotion
# #     np.save(featurefile, mfcc_feat)
#
#
# os.chdir("/home/smu/Desktop/RNN/audiodata/emo_sixseconds")
#
# print("Generating features from emoDB ...")
#
# for aud in tqdm(glob.glob("*.wav")):
#     (rate,sig) = wav.read(aud)
#     mfcc_feat = mfcc(sig, rate, winlen=WINDOW_SIZE, winstep=WINDOW_STEP, nfft=NFFT, preemph=PREEMPH)
#     emotion = "N"
#     if "W" in aud:
#         emotion = "W"
#     elif "L" in aud:
#         emotion = "L"
#     elif "E" in aud:
#         emotion = "E"
#     elif "A" in aud:
#         emotion = "A"
#     elif "F" in aud:
#         emotion = "F"
#     elif "T" in aud:
#         emotion = "T"
#     featurefile = "../../train_data/" + aud + "_" + emotion
#     np.save(featurefile, mfcc_feat)
#
# # os.chdir("/home/smu/Desktop/RNN/audiodata/emo_sixseconds_envnoise")
# #
# # print("Generating features from emoDB with noise ...")
# #
# # for aud in tqdm(glob.glob("*.wav")):
# #     (rate,sig) = wav.read(aud)
# #     mfcc_feat = mfcc(sig, rate, winlen=WINDOW_SIZE, winstep=WINDOW_STEP, nfft=NFFT, preemph=PREEMPH)
# #     emotion = "N"
# #     if "W" in aud:
# #         emotion = "W"
# #     elif "L" in aud:
# #         emotion = "L"
# #     elif "E" in aud:
# #         emotion = "E"
# #     elif "A" in aud:
# #         emotion = "A"
# #     elif "F" in aud:
# #         emotion = "F"
# #     elif "T" mixednoisein aud:
# #         emotion = "T"
# #     featurefile = "../../train_data/" + aud + "___" + emotion
# #     np.save(featurefile, mfcc_feat)
#
# os.chdir("/home/smu/Desktop/RNN/audiodata/zenodo_sixseconds")
#
# print("Generating features from zenodo-database...")
#
# for aud in tqdm(glob.glob("*.wav")):
#     (rate,sig) = wav.read(aud)
#     mfcc_feat = mfcc(sig, rate, winlen=WINDOW_SIZE, winstep=WINDOW_STEP, nfft=NFFT, preemph=PREEMPH)
#     emotion = "N"
#     if "W" in aud:
#         emotion = "W"
#     elif "L" in aud:
#         emotion = "L"
#     elif "E" in aud:
#         emotion = "E"
#     elif "A" in aud:
#         emotion = "A"
#     elif "F" in aud:
#         emotion = "F"
#     elif "T" in aud:
#         emotion = "T"
#     featurefile = "../../train_data/" + aud + "_" + emotion
#     np.save(featurefile, mfcc_feat)
#
# # os.chdir("/home/smu/Desktop/RNN/audiodata/zenodo_sixseconds_envnoise")
# #
# # print("Generating features from zenodo-database with noise...")
# #
# # for aud in tqdm(glob.glob("*.wav")):
# #     (rate,sig) = wav.read(aud)
# #     mfcc_feat = mfcc(sig, rate, winlen=WINDOW_SIZE, winstep=WINDOW_STEP, nfft=NFFT, preemph=PREEMPH)
# #     emotion = "N"
# #     if "W" in aud:
# #         emotion = "W"
# #     elif "L" in aud:
# #         emotion = "L"
# #     elif "E" in aud:
# #         emotion = "E"
# #     elif "A" in aud:
# #         emotion = "A"
# #     elif "F" in aud:
# #         emotion = "F"
# #     elif "T" in aud:
# #         emotion = "T"
# #     featurefile = "../../train_data/" + aud + "___" + emotion
# #     np.save(featurefile, mfcc_feat)

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

print("Chosing two validation samples for each emotion ...")

os.chdir(PATH_VALIDATIONDATA)

try:
    os.mkdir(MODELNAME)
except Exception as e:
    print('Failed to create model folder: %s' % (e))

c = 0
while c != 2:
    random_file = random.choice(os.listdir(PATH_TRAINDATA))
    if "N" in random_file:
        src = PATH_TRAINDATA + random_file
        dest = PATH_VALIDATIONDATA + MODELNAME + "/" + random_file
        shutil.move(src,dest)
        c += 1

c = 0
while c != 2:
    random_file = random.choice(os.listdir(PATH_TRAINDATA))
    if "W" in random_file:
        src = PATH_TRAINDATA + random_file
        dest = PATH_VALIDATIONDATA + MODELNAME + "/" + random_file
        shutil.move(src,dest)
        c += 1

c = 0
while c != 2:
    random_file = random.choice(os.listdir(PATH_TRAINDATA))
    if "L" in random_file:
        src = PATH_TRAINDATA + random_file
        dest = PATH_VALIDATIONDATA + MODELNAME + "/" + random_file
        shutil.move(src,dest)
        c += 1

c = 0
while c != 2:
    random_file = random.choice(os.listdir(PATH_TRAINDATA))
    if "E" in random_file:
        src = PATH_TRAINDATA + random_file
        dest = PATH_VALIDATIONDATA + MODELNAME + "/" + random_file
        shutil.move(src,dest)
        c += 1

c = 0
while c != 2:
    random_file = random.choice(os.listdir(PATH_TRAINDATA))
    if "A" in random_file:
        src = PATH_TRAINDATA + random_file
        dest = PATH_VALIDATIONDATA + MODELNAME + "/" + random_file
        shutil.move(src,dest)
        c += 1

c = 0
while c != 2:
    random_file = random.choice(os.listdir(PATH_TRAINDATA))
    if "F" in random_file:
        src = PATH_TRAINDATA + random_file
        dest = PATH_VALIDATIONDATA + MODELNAME + "/" + random_file
        shutil.move(src,dest)
        c += 1

c = 0
while c != 2:
    random_file = random.choice(os.listdir(PATH_TRAINDATA))
    if "T" in random_file:
        src = PATH_TRAINDATA + random_file
        dest = PATH_VALIDATIONDATA + MODELNAME + "/" + random_file
        shutil.move(src,dest)
        c += 1

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

try:
    filepath = PATH_VALIDATIONDATA + MODELNAME + "/"
    files = os.listdir(filepath)
    for f in files:
        shutil.copy2(filepath + f, PATH_TRAINDATA)
except Exception as e:
    print(e)

print("Generating model ...")

model = tf.keras.Sequential()
# model.add(layers.LSTM((512), input_shape=(None, 13), return_sequences=True))
# model.add(layers.Dropout(0.6))
# model.add(layers.LSTM((512), input_shape=(None, 13), return_sequences=True))
# model.add(layers.Dropout(0.5))
# model.add(layers.LSTM((512), input_shape=(None, 13), return_sequences=True))
# model.add(layers.Dropout(0.4))
model.add(layers.LSTM((512), input_shape=(None, 13)))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(7, activation='softmax'))

rms = tf.keras.optimizers.RMSprop(learning_rate=0.001)

model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'])
model.summary()

print("Clearing weights folder ... ")

for filename in os.listdir(PATH_WEIGHTS):
    file_path = os.path.join(PATH_WEIGHTS, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

print("Training ...")

os.chdir("/home/smu/Desktop/RNN")
log_dir = "logs/" + MODELNAME
weights_dir = "temp/"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=weights_dir,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

model.fit(features_train, ltr, epochs=50, batch_size=128, validation_data=(features_test, ltt), callbacks=[tensorboard_callback, model_checkpoint_callback])

model.load_weights(weights_dir)

model_dir = 'models/' + MODELNAME
model.save(model_dir)

print("Model trained and saved!")

# Eine Studie über Emotionserkennung mithilfe menschlicher Stimme mit rekurrenten neuronalen Netzen
# A study on emotion recognition using human voice with Recurrent Neural Networks
