# Stats MFCC Calculation

### Samplerate
SAMPLERATE = 48000

Samplerate is given from the audio source

### Window-Size
WINDOW_SIZE = 0.8 s

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

After testing, we found out that a LSTM-RESNET produces the best results (average 77 % validation accuracy):

```python
input1 = layers.Input(shape=(None, 13))
lstm1 = layers.LSTM(UNITS, return_sequences=True)(input1)
lstm2 = layers.LSTM(UNITS, return_sequences=True)(lstm1)
merge1 = layers.Concatenate(axis=2)([input1,lstm2])
merge2 = layers.Concatenate(axis=2)([input1,merge1])
lstm3 = layers.LSTM(UNITS, return_sequences=True)(merge2)
lstm4 = layers.LSTM(UNITS, return_sequences=True)(lstm3)
merge3 = layers.Concatenate(axis=2)([merge1,lstm4])
merge4 = layers.Concatenate(axis=2)([input1,merge3])
lstm5 = layers.LSTM(UNITS, return_sequences=True)(merge4)
lstm6 = layers.LSTM(UNITS, return_sequences=True)(lstm5)
merge5 = layers.Concatenate(axis=2)([merge3,lstm6])
merge6 = layers.Concatenate(axis=2)([input1,merge5])
lstm7 = layers.LSTM(UNITS, return_sequences=True)(merge6)
lstm8 = layers.LSTM(UNITS, return_sequences=True)(lstm7)
merge7 = layers.Concatenate(axis=2)([merge5,lstm8])
merge8 = layers.Concatenate(axis=2)([input1,merge7])
lstm9 = layers.LSTM(UNITS)(merge8)
dropout1 = layers.Dropout(0.4)(lstm9)
dense1 = layers.Dense(7, activation='softmax')(dropout1)
model = Model(inputs=input1, outputs=dense1)
```
