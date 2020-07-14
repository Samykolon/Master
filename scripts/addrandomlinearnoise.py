from pydub import AudioSegment
from pydub.generators import WhiteNoise
from pydub.generators import Sine
from pydub.utils import make_chunks
import os
import glob, shutil, sys, random

def set_to_target_level(sound, target_level):
    difference = target_level - sound.dBFS
    return sound.apply_gain(difference)

os.chdir("/home/smu/Desktop/RNN/audiodata/zenodo_sixseconds")

noisefolder = "/home/smu/Desktop/RNN/audiodata/zenodo_sixseconds_whitenoise/"

print("Adding white noise to files ...")

for aud in glob.glob("*.wav"):
    myaudio = AudioSegment.from_file(aud, "wav")
    noise = WhiteNoise().to_audio_segment(duration=len(myaudio))
    noise = set_to_target_level(noise, -92.0)
    myaudio = set_to_target_level(myaudio, 24.0)
    combined = myaudio.overlay(noise)
    filepath = noisefolder + aud
    combined.export(filepath, format="wav")

print("Done")
