import pyaudio
import wave
import os
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 48000
CHUNK = 1000
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "/home/smu/Desktop/RNN/own/freude.wav"
WAVE_OUTPUT_DOWN = "/home/smu/Desktop/RNN/own/F1_16000.wav"
WAVE_TEST = "/home/smu/Desktop/RNN/speakers_short/03a01Fa.wavshort.wav"

INPUT_DEVICE = "AT2020 USB: Audio (hw:2,0)" # Name of the input device

# Lookup the index of the desired Input-Device, make sure jack is running
pa = pyaudio.PyAudio()
chosen_device_index = -1
for x in range(0,pa.get_device_count()):
    info = pa.get_device_info_by_index(x)
    # print (pa.get_device_info_by_index(x))
    if info["name"] == INPUT_DEVICE:
        chosen_device_index = info["index"]

print ("Chosen index: ", chosen_device_index)
pa.terminate()


audio = pyaudio.PyAudio()

stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=CHUNK, input_device_index=chosen_device_index)

frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

stream.stop_stream()
stream.close()
audio.terminate()

waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
waveFile.setnchannels(CHANNELS)
waveFile.setsampwidth(audio.get_sample_size(FORMAT))
waveFile.setframerate(RATE)
waveFile.writeframes(b''.join(frames))
waveFile.close()
