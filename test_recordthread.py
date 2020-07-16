import pyaudio
import wave
import os
import audioop
import glob, shutil, sys, random
from pydub import AudioSegment
from pydub.utils import make_chunks
import soundfile as sf

FORMAT = pyaudio.paInt16
OUTCHANNELS = 1
CHANNELS = 2
OUTRATE = 48000
RATE = 48000
CHUNK = 1000
RECORD_SECONDS = 240
WAVE_OUTPUT_FILENAME = "/home/smu/Desktop/RNN/TEST.wav"
INPUT_DEVICE = "AT2020 USB: Audio (hw:1,0)" # Name of the input device

def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

pa = pyaudio.PyAudio()
chosen_device_index = -1
for x in range(0,pa.get_device_count()):
    info = pa.get_device_info_by_index(x)
    print (pa.get_device_info_by_index(x))
    if info["name"] == INPUT_DEVICE:
        chosen_device_index = info["index"]

print ("Chosen index: ", chosen_device_index)
pa.terminate()

print("Start Recording")

audio = pyaudio.PyAudio()

stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=CHUNK, input_device_index=chosen_device_index)

frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK, exception_on_overflow = False)
    frames.append(data)
    if len(frames) == (RATE / CHUNK * 6):
        waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(frames))
        waveFile.close()
        frames = []
        sound = AudioSegment.from_file(WAVE_OUTPUT_FILENAME)
        sound = sound.set_frame_rate(OUTRATE)
        sound = sound.set_channels(OUTCHANNELS)
        sound = match_target_amplitude(sound, -40.0)
        sound.export(WAVE_OUTPUT_FILENAME, format="wav")

print("Done Recording")

stream.stop_stream()
stream.close()
audio.terminate()
