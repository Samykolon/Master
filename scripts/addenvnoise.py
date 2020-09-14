# (C) Samuel Dressel 2020
# This script will add a random "real" noise to every audio sample (overlay)

from pydub import AudioSegment
from pydub.utils import make_chunks
import os
import glob, shutil, sys, random

PATH_ENVNOISE = "/home/smu/Desktop/RNN/audiodata/envnoise_sixseconds/"
PATH_OUTPUT = "/home/smu/Desktop/RNN/audiodata/own_sixseconds_envnoise/"
PATH_INPUT = "/home/smu/Desktop/RNN/audiodata/own_sixseconds"

# Function to normalize set the overlay gain of the noise
def set_to_target_level(sound, target_level):
    difference = target_level - sound.dBFS
    return sound.apply_gain(difference)

os.chdir(PATH_INPUT)

print("Adding random real noise ...")

for aud in glob.glob("*.wav"):
    org_file = AudioSegment.from_file(aud , "wav")
    random_file = random.choice(os.listdir(PATH_ENVNOISE))
    noise_file = AudioSegment.from_file(PATH_ENVNOISE + random_file, "wav")
    filepath = PATH_OUTPUT + aud
    combined = org_file.overlay(noise_file)
    combined.export(filepath, format="wav")

print("Done!")
