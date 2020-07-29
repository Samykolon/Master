# (C) Samuel Dressel 2020
# Make all samples of the zenodo dataset 6 seconds long

import pyaudio
import wave
import os
import glob, shutil, sys, random
import numpy as np
from pydub import AudioSegment
from pydub.utils import make_chunks

FORMAT = pyaudio.paInt16
RATE = 48000
CHANNELS = 2

RECORD_SECONDS = 6
RECORD_FRAMES = RECORD_SECONDS * RATE

os.chdir("/home/smu/Desktop/RNN/zenodo_long")

print("Making each sample 6 seconds long")

for aud in glob.glob("*.wav"):
    final = AudioSegment.empty()
    file = AudioSegment.from_file(aud , "wav")
    framecount = file.frame_count()
    number_full_wav = RECORD_FRAMES // framecount
    i = 0.0
    while i < (number_full_wav + 1.0):
        final += file
        i += 1.0
    final = final[:6000]
    filepath = "/home/smu/Desktop/RNN/zenodo_sixseconds/6_" + aud
    file_handle = final.export(filepath, format="wav")

print("Done")
