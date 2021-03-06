# (C) Samuel Dressel 2020
# Train a 9-LSTM-RESNET with a 40-MFCC-Feature + 8-Time-Spectral-Feature dataset (48 Features total)

import frequencyfeatures
from pyAudioAnalysis import audioBasicIO
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
from tensorflow.keras.models import Model

import sklearn.metrics
import matplotlib.pyplot as plt
import itertools
import io

# Preemph-Filter to reduce noise
PREEMPH = 0.0

# Number of Testsamples
NUMBER_TESTSAMPLES = 200

# Name of the model (for saving and logs)
PREMODELNAME = "rnn_full_mfcc+time+spec_nopreemph_mixednoise2000_resnet_ws08_512_"

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

# Path where the train-data is stored
PATH_TRAINDATA = "/home/smu/Desktop/RNN/train_data/"
# Path where the test-data is stored - gets randomly picked out of traindata
PATH_TESTDATA = "/home/smu/Desktop/RNN/test_data/"
# Path for the validation_data for later testing
PATH_VALIDATIONDATA = "/home/smu/Desktop/RNN/validation_data/"
# Path for the temporal saved weights
PATH_WEIGHTS = "/home/smu/Desktop/RNN/temp/"
# class_names
CLASSNAMES = ['Wut', 'Langeweile', 'Ekel', 'Angst', 'Freude', 'Trauer', 'Neutral']

os.chdir("/home/smu/Desktop/RNN/audiodata/own_sixseconds")

print("Generating features from own recordings ...")

for aud in tqdm(glob.glob("*.wav")):
    [Fs, x] = audioBasicIO.read_audio_file(aud)
    F, f_names = frequencyfeatures.feature_extraction(x, Fs, WINDOW_SIZE*Fs, WINDOW_STEP*Fs)
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
    featurefile = "../../train_data/" + aud + "_" + emotion
    np.save(featurefile, F)

os.chdir("/home/smu/Desktop/RNN/audiodata/own_sixseconds_envnoise")

print("Generating features from own recordings with noise ...")

for aud in tqdm(glob.glob("*.wav")):
    [Fs, x] = audioBasicIO.read_audio_file(aud)
    F, f_names = frequencyfeatures.feature_extraction(x, Fs, WINDOW_SIZE*Fs, WINDOW_STEP*Fs)
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
    featurefile = "../../train_data/" + aud + "_" + emotion
    np.save(featurefile, F)

os.chdir("/home/smu/Desktop/RNN/audiodata/emo_sixseconds")

print("Generating features from emoDB ...")

for aud in tqdm(glob.glob("*.wav")):
    [Fs, x] = audioBasicIO.read_audio_file(aud)
    F, f_names = frequencyfeatures.feature_extraction(x, Fs, WINDOW_SIZE*Fs, WINDOW_STEP*Fs)
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
    featurefile = "../../train_data/" + aud + "___" + emotion
    np.save(featurefile, F)

os.chdir("/home/smu/Desktop/RNN/audiodata/emo_sixseconds_envnoise")

print("Generating features from emoDB with noise ...")

for aud in tqdm(glob.glob("*.wav")):
    [Fs, x] = audioBasicIO.read_audio_file(aud)
    F, f_names = frequencyfeatures.feature_extraction(x, Fs, WINDOW_SIZE*Fs, WINDOW_STEP*Fs)
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
    featurefile = "../../train_data/" + aud + "_" + emotion
    np.save(featurefile, F)

os.chdir("/home/smu/Desktop/RNN/audiodata/zenodo_sixseconds")

print("Generating features from zenodo-database ...")

for aud in tqdm(glob.glob("*.wav")):
    [Fs, x] = audioBasicIO.read_audio_file(aud)
    F, f_names = frequencyfeatures.feature_extraction(x, Fs, WINDOW_SIZE*Fs, WINDOW_STEP*Fs)
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
    featurefile = "../../train_data/" + aud + "_" + emotion
    np.save(featurefile, F)

os.chdir("/home/smu/Desktop/RNN/audiodata/zenodo_sixseconds_envnoise")

print("Generating features from zenodo-database with noise ...")

for aud in tqdm(glob.glob("*.wav")):
    [Fs, x] = audioBasicIO.read_audio_file(aud)
    F, f_names = frequencyfeatures.feature_extraction(x, Fs, WINDOW_SIZE*Fs, WINDOW_STEP*Fs)
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
    featurefile = "../../train_data/" + aud + "___" + emotion
    np.save(featurefile, F)

cc = 1

while cc < 6:

    MODELNAME = PREMODELNAME + str(cc)

    # Clear test_data folder an move random files from the train_data folder in
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

    # Create folder inside the validation_data folder for this specific model and move four files for each emotion in
    print("Chosing four validation samples for each emotion ...")

    os.chdir(PATH_VALIDATIONDATA)

    try:
        os.mkdir(MODELNAME)
    except Exception as e:
        print('Failed to create model folder: %s' % (e))

    c = 0
    while c != 4:
        random_file = random.choice(os.listdir(PATH_TRAINDATA))
        if "N" in random_file:
            src = PATH_TRAINDATA + random_file
            dest = PATH_VALIDATIONDATA + MODELNAME + "/" + random_file
            shutil.move(src,dest)
            c += 1

    c = 0
    while c != 4:
        random_file = random.choice(os.listdir(PATH_TRAINDATA))
        if "W" in random_file:
            src = PATH_TRAINDATA + random_file
            dest = PATH_VALIDATIONDATA + MODELNAME + "/" + random_file
            shutil.move(src,dest)
            c += 1

    c = 0
    while c != 4:
        random_file = random.choice(os.listdir(PATH_TRAINDATA))
        if "L" in random_file:
            src = PATH_TRAINDATA + random_file
            dest = PATH_VALIDATIONDATA + MODELNAME + "/" + random_file
            shutil.move(src,dest)
            c += 1

    c = 0
    while c != 4:
        random_file = random.choice(os.listdir(PATH_TRAINDATA))
        if "E" in random_file:
            src = PATH_TRAINDATA + random_file
            dest = PATH_VALIDATIONDATA + MODELNAME + "/" + random_file
            shutil.move(src,dest)
            c += 1

    c = 0
    while c != 4:
        random_file = random.choice(os.listdir(PATH_TRAINDATA))
        if "A" in random_file:
            src = PATH_TRAINDATA + random_file
            dest = PATH_VALIDATIONDATA + MODELNAME + "/" + random_file
            shutil.move(src,dest)
            c += 1

    c = 0
    while c != 4:
        random_file = random.choice(os.listdir(PATH_TRAINDATA))
        if "F" in random_file:
            src = PATH_TRAINDATA + random_file
            dest = PATH_VALIDATIONDATA + MODELNAME + "/" + random_file
            shutil.move(src,dest)
            c += 1

    c = 0
    while c != 4:
        random_file = random.choice(os.listdir(PATH_TRAINDATA))
        if "T" in random_file:
            src = PATH_TRAINDATA + random_file
            dest = PATH_VALIDATIONDATA + MODELNAME + "/" + random_file
            shutil.move(src,dest)
            c += 1

    # Initialising GPU by setting memory_growth to true
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

    # Empty arrays for train and test data
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

    # After generting the tensors, we can move the files from the validation_data folder
    # and from the test_data folder back so we can use them for a new session
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

    valacc = 0.0

    # Generating the model
    print("Generating model ...")

    # RESNET
    input1 = layers.Input(shape=(None, 48))
    lstm1 = layers.LSTM(UNITS, return_sequences=True)(input1)
    lstm2 = layers.LSTM(UNITS, return_sequences=True)(lstm1)
    merge1 = layers.Concatenate(axis=2)([input1,lstm2])
    merge2 = layers.Concatenate(axis=2)([input1,merge1])
    lstm3 = layers.LSTM(UNITS, return_sequences=True)(merge2)
    lstm4 = layers.LSTM(UNITS, return_sequences=True)(lstm3)
    merge3 = layers.Concatenate(axis=2)([merge1,lstm4])
    merge4 = layers.Concatenate(axis=2)([input1,merge3])
    lstm5 = layers.LSTM(UNITS, return_sequences=True)(merge4)
    lstm6 = layers.LSTM(UNITS, return_sequences=True)(lstm5)
    merge5 = layers.Concatenate(axis=2)([merge3,lstm6])
    merge6 = layers.Concatenate(axis=2)([input1,merge5])
    lstm7 = layers.LSTM(UNITS, return_sequences=True)(merge6)
    lstm8 = layers.LSTM(UNITS, return_sequences=True)(lstm7)
    merge7 = layers.Concatenate(axis=2)([merge5,lstm8])
    merge8 = layers.Concatenate(axis=2)([input1,merge7])
    lstm9 = layers.LSTM(UNITS)(merge8)
    dropout1 = layers.Dropout(0.4)(lstm9)
    dense1 = layers.Dense(7, activation='softmax')(dropout1)
    model = Model(inputs=input1, outputs=dense1)

    rms = tf.keras.optimizers.RMSprop(learning_rate=0.001)

    model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'])
    model.summary()

    # While fitting the model only the best weights get saved so we clear the weight-cache
    # from previous training
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

    # Next three functions are for generating the confusion matrix and store it as part of tensorboard
    def plot_to_image(figure):
      # Save the plot to a PNG in memory.
      buf = io.BytesIO()
      plt.savefig(buf, format='png')
      # Closing the figure prevents it from being displayed directly inside
      # the notebook.
      plt.close(figure)
      buf.seek(0)
      # Convert PNG buffer to TF image
      image = tf.image.decode_png(buf.getvalue(), channels=4)
      # Add the batch dimension
      image = tf.expand_dims(image, 0)
      return image

    def plot_confusion_matrix(cm, class_names):
      figure = plt.figure(figsize=(8, 8))
      plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
      plt.title("Confusion matrix")
      plt.colorbar()
      tick_marks = np.arange(len(class_names))
      plt.xticks(tick_marks, class_names, rotation=45)
      plt.yticks(tick_marks, class_names)

      # Normalize the confusion matrix.
      cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

      # Use white text if squares are dark; otherwise black.
      threshold = cm.max() / 2.
      for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)


      plt.ylabel('True label')
      plt.xlabel('Predicted label')
      plt.tight_layout()
      return figure

    file_writer_cm = tf.summary.create_file_writer(log_dir + '/cm')

    def log_confusion_matrix(epoch, logs):
      global valacc
      current = logs.get("val_accuracy")
      if current > valacc:
          valacc = current
          # Use the model to predict the values from the validation dataset.
          test_pred = np.argmax(model.predict(features_test), axis=-1)
          test_real = np.argmax(ltt, axis=1)
          # Calculate the confusion matrix.
          cm = sklearn.metrics.confusion_matrix(test_real, test_pred)
          # Log the confusion matrix as an image summary.
          figure = plot_confusion_matrix(cm, class_names=CLASSNAMES)
          cm_image = plot_to_image(figure)

          # Log the confusion matrix as an image summary.
          with file_writer_cm.as_default():
              tf.summary.image("Confusion Matrix", cm_image, step=epoch)

    # First callback; generate confusion matrix while training
    cm_callback = keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)

    # Second callback, generate tensorboard logs while training
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Third callback; save the best model-weights (best validation_accuracy)
    weights_dir = "temp/"
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=weights_dir,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    model.fit(features_train, ltr, epochs=50, batch_size=128, validation_data=(features_test, ltt), callbacks=[tensorboard_callback, model_checkpoint_callback, cm_callback])

    # Load the temporarly saved best model weights and save the entire model with this weights
    model.load_weights(weights_dir)

    model_dir = 'models/' + MODELNAME
    model.save(model_dir)

    print("Model trained and saved!")

    cc = cc + 1
