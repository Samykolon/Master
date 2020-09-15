from moviepy.editor import *
import os
import glob, shutil, sys, random

PATH = "/home/smu/Desktop/RNN/audiodata/ValidationSET_AVI"

os.chdir(PATH)

for aud in glob.glob("*.avi"):
    audioclip = AudioFileClip(aud)
    AudioFileClip.write_audiofile(audioclip, "/home/smu/Desktop/RNN/audiodata/ValidationSET_WAV/" + aud + ".wav")
