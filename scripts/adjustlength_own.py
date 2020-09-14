# (C) Samuel Dressel 2020
# Make all samples of the own recordings 6 seconds long

from pydub import AudioSegment
from pydub.utils import make_chunks
import os
import glob, shutil, sys, random

os.chdir("/home/smu/Desktop/RNN/audiodata/0_srcfiles/own")

print("Making each sample 6 seconds long")

for aud in glob.glob("*.wav"):
    myaudio = AudioSegment.from_file(aud, "wav")
    chunk_length_ms = 6000 # pydub calculates in millisec
    chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec
    for i, chunk in enumerate(chunks):
        c_name = "_{0}.wav".format(i)
        filepath = "/home/smu/Desktop/RNN/audiodata/own_sixseconds/" + aud
        chunk_name = filepath + c_name
        chunk.export(chunk_name, format="wav")

print("Done")
