# (C) Samuel Dressel
# Script to make the channelnumber and the samplerate for all the different dataset
# sources the same. This script also normalizes the gain of each sample.
# The commented lines in the script are for controlling the parameters by printing them out
# for each sample
import wave
import audioop
import glob, shutil, sys, random, os
from pydub import AudioSegment
from pydub.utils import make_chunks
import soundfile as sf

PATH_EMO = "/home/smu/Desktop/RNN/audiodata/emo_sixseconds/"
PATH_EMO_NEW = "/home/smu/Desktop/RNN/audiodata/emo_sixseconds/"
PATH_ZENODO = "/home/smu/Desktop/RNN/audiodata/zenodo_sixseconds/"
PATH_ZENODO_NEW = "/home/smu/Desktop/RNN/audiodata/zenodo_sixseconds/"
PATH_OWN = "/home/smu/Desktop/RNN/audiodata/own_sixseconds/"
PATH_OWN_NEW = "/home/smu/Desktop/RNN/audiodata/own_sixseconds/"
PATH_ENV = "/home/smu/Desktop/RNN/audiodata/envnoise_sixseconds/"
PATH_ENV_NEW = "/home/smu/Desktop/RNN/audiodata/envnoise_sixseconds/"
OUTCHANNELS = 1
OUTRATE = 48000

def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

os.chdir(PATH_EMO)

for aud in glob.glob("*.wav"):
    sound = AudioSegment.from_file(aud)
    sound = sound.set_frame_rate(OUTRATE)
    sound = sound.set_channels(OUTCHANNELS)
    silence = AudioSegment.silent(duration=0.1)  #duration in milliseconds
    silence.set_frame_rate(OUTRATE)
    silence.set_channels(OUTCHANNELS)
    sound += silence
    sound += silence
    sound = match_target_amplitude(sound, -40.0)
    filepath = PATH_EMO_NEW + aud
    sound.export(filepath, format="wav")

# os.chdir(PATH_EMO_NEW)
#
# for aud in glob.glob("*.wav"):
#     ob = sf.SoundFile(aud)
#     print('Sample rate: {}'.format(ob.samplerate))
#     print('Channels: {}'.format(ob.channels))
#     print('Subtype: {}'.format(ob.subtype))
#     print('Frames: {}'.format(ob.frames))

os.chdir(PATH_OWN)

for aud in glob.glob("*.wav"):
    sound = AudioSegment.from_file(aud)
    sound = sound.set_frame_rate(OUTRATE)
    sound = sound.set_channels(OUTCHANNELS)
    sound = match_target_amplitude(sound, -40.0)
    filepath = PATH_OWN_NEW + aud
    sound.export(filepath, format="wav")

# os.chdir(PATH_OWN_NEW)
#
# for aud in glob.glob("*.wav"):
#     ob = sf.SoundFile(aud)
#     print('Sample rate: {}'.format(ob.samplerate))
#     print('Channels: {}'.format(ob.channels))
#     print('Subtype: {}'.format(ob.subtype))
#     print('Frames: {}'.format(ob.frames))

os.chdir(PATH_ZENODO)

for aud in glob.glob("*.wav"):
    sound = AudioSegment.from_file(aud)
    sound = sound.set_frame_rate(OUTRATE)
    sound = sound.set_channels(OUTCHANNELS)
    sound = match_target_amplitude(sound, -40.0)
    filepath = PATH_ZENODO_NEW + aud
    sound.export(filepath, format="wav")

# os.chdir(PATH_ZENODO_NEW)
#
# for aud in glob.glob("*.wav"):
#     ob = sf.SoundFile(aud)
#     print('Sample rate: {}'.format(ob.samplerate))
#     print('Channels: {}'.format(ob.channels))
#     print('Subtype: {}'.format(ob.subtype))
#     print('Frames: {}'.format(ob.frames))

os.chdir(PATH_ENV)

for aud in glob.glob("*.wav"):
    sound = AudioSegment.from_file(aud)
    sound = sound.set_frame_rate(OUTRATE)
    sound = sound.set_channels(OUTCHANNELS)
    silence = AudioSegment.silent(duration=0.1)  #duration in milliseconds
    silence.set_frame_rate(OUTRATE)
    silence.set_channels(OUTCHANNELS)
    sound += silence
    sound += silence
    sound += silence
    silence = AudioSegment.silent(duration=833.4)  #duration in milliseconds
    silence.set_frame_rate(OUTRATE)
    silence.set_channels(OUTCHANNELS)
    sound += silence
    sound = match_target_amplitude(sound, -40.0)
    filepath = PATH_ENV_NEW + aud
    sound.export(filepath, format="wav")

# os.chdir(PATH_ENV_NEW)
#
# for aud in glob.glob("*.wav"):
#     ob = sf.SoundFile(aud)
#     print('Sample rate: {}'.format(ob.samplerate))
#     print('Channels: {}'.format(ob.channels))
#     print('Subtype: {}'.format(ob.subtype))
#     print('Frames: {}'.format(ob.frames))
