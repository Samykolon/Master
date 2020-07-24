# Stats MFCC Calculation

### Samplerate
SAMPLERATE = 48000

Samplerate is given from the audio source

### Window-Size
WINDOW_SIZE = 0.8 s

https://arxiv.org/pdf/1701.08071.pdf - Emotion Recognition From Speech With RecurrentNeural Networks

30 milliseconds roughly correspond to the duration of one phoneme in the normal flow of spoken English.
200 milliseconds is the approximate duration of one word.
Experiments do not show significant difference in terms of quality.  But computation time rises with the reduction in frameduration due to bigger number of frames.

### NFFT
NFFT = 65536

By default, the FFT size is the first equal or superior power of 2 of the window size. If we have a samplerate of 48000 Hz and a window size of 800 ms, we get 38400 samples in each window. The next superior power would be 65536 so we choose that.

### Window-Stepsize
WINDOW_STEP = 0.1 s

Window-Duration/8 - Overlapping Parameter

### Signallength
SIGNALLENGTH = 6 s

From the paper: Attentive Convolutional Neural Network based Speech Emotion Recognition:
A Study on the Impact of Input Features, Signal Length, and Acted Speech we can retrieve 6 s as an optimal Length

# Stats for the models

### Model-Structure

After testing, we found out that a 5-LSTM produces the best results (average 72.3 % validation accuracy):

```python
model = tf.keras.Sequential()
model.add(layers.LSTM((UNITS), input_shape=(None, 13), return_sequences=True))
model.add(layers.LSTM((UNITS), input_shape=(None, 13), return_sequences=True))
model.add(layers.LSTM((UNITS), input_shape=(None, 13), return_sequences=True))
model.add(layers.LSTM((UNITS), input_shape=(None, 13), return_sequences=True))
model.add(layers.LSTM((UNITS), input_shape=(None, 13)))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(UNITS, activation='relu'))
model.add(layers.Dense(7, activation='softmax'))
```
