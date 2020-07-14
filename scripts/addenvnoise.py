from pydub import AudioSegment
from pydub.utils import make_chunks
import os
import glob, shutil, sys, random

RATE = 16000
RECORD_SECONDS = 6
RECORD_FRAMES = RECORD_SECONDS * RATE
PATH_ENVNOISE = "/home/smu/Desktop/RNN/audiodata/envnoise/"

noisefolder = "/home/smu/Desktop/RNN/audiodata/emo_sixseconds_envnoise/"

def set_to_target_level(sound, target_level):
    difference = target_level - sound.dBFS
    return sound.apply_gain(difference)

os.chdir("/home/smu/Desktop/RNN/audiodata/emo_sixseconds")

print("Adding random real noise ...")

for aud in glob.glob("*.wav"):
    final = AudioSegment.empty()
    org_file = AudioSegment.from_file(aud , "wav")
    random_file = random.choice(os.listdir(PATH_ENVNOISE))
    noise_file = AudioSegment.from_file(PATH_ENVNOISE + random_file, "wav")
    framecount = noise_file.frame_count()
    number_full_wav = RECORD_FRAMES // framecount
    i = 0.0
    while i < (number_full_wav + 1.0):
        final += noise_file
        i += 1.0
    final = final[:6000]
    final = set_to_target_level(final, -64.0)
    combined = org_file.overlay(final)
    filepath = noisefolder + aud
    combined.export(filepath, format="wav")

print("Done!")
