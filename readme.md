# Stats MFCC Calculation

### Samplerate
SAMPLERATE = 16000

Samplerate is given from the audio source

### Window-Size
WINDOW_SIZE = 0.032 s

How big is the window features get generated from - find source!

### NFFT
NFFT = 512

By default, the FFT size is the first equal or superior power of 2 of the window size. If we have a samplerate of 16000 Hz and a window size of 32 ms, we get 512 samples in each window. The next superior power would be 512 so we choose that.

### Window-Stepsize
WINDOW_STEP = 0.004 s

Window-Duration/8 - Overlapping Parameter

### Signallength
SIGNALLENGTH = 6 s

From the paper: Attentive Convolutional Neural Network based Speech Emotion Recognition:A Study on the Impact of Input Features, Signal Length, and Acted Speech we can retrieve 6 s as an optimal Length
