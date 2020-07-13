# Stats MFCC Calculation

### Samplerate
SAMPLERATE = 16000

Samplerate is given from the audio source

### Window-Size
WINDOW_SIZE = 0.032 s

How big is the window features get generated from - find source!

https://arxiv.org/pdf/1701.08071.pdf - Emotion Recognition From Speech With RecurrentNeural Networks

30 milliseconds roughly correspond to the duration of one phoneme in the normal flow of spoken English.
200 milliseconds is the approximate duration of one word.
Experiments do not show significant difference in terms of quality.  But computation time rises with the reduction in frameduration due to bigger number of frames.

### NFFT
NFFT = 512

By default, the FFT size is the first equal or superior power of 2 of the window size. If we have a samplerate of 16000 Hz and a window size of 32 ms, we get 512 samples in each window. The next superior power would be 512 so we choose that.

### Window-Stepsize
WINDOW_STEP = 0.004 s

Window-Duration/8 - Overlapping Parameter

### Signallength
SIGNALLENGTH = 6 s

From the paper: Attentive Convolutional Neural Network based Speech Emotion Recognition:A Study on the Impact of Input Features, Signal Length, and Acted Speech we can retrieve 6 s as an optimal Length

# Testen


1. Macht es einen Unterschied, nur deutsche Aufnahmen, nur englische Aufnahmen oder gemischte Aufnahmen als Input zu verwenden?
2. Welche Features werden genommen (momentan nur MFCCs) - lohnt es sich auch Features wie Time-Domain Features (zero crossing rate, energy, entropy of energy) und Spectral-Domain-Features (Spectral centroid, spectral spread, spectral entropy, spectral flux, spectral rollof) mit aufzunehmen?
3. Was funktioniert besser - das Netzwerk mit vorherigem Preemph-Filter oder ohne aufzunehmen?
4. Welchen Unterschied machen die benutzen Layer - 1, 2, oder 3 LSTM-Layer? Oder mit GRU versuchen?

Testvorgänge:

1. Features und Noise im  Datensatz

|                      | MFCC mit Preemph | MFCC ohne Preemph | MFCC mit Preemph und Spectral/Time-Features | MFCC ohne Preemph mit Spectral/Time Features |
|----------------------|------------------|-------------------|---------------------------------------------|----------------------------------------------|
| **Testdaten mit Noise**  |                  |                   |                                             |                                              |
| **Testdaten ohne Noise** |                  |                   |                                             |                                              |
| **Testdaten mit echter Noise** |                  |                   |                                             |                                              |

2. Beste Ergebnisse nehmen und deutsch, englisch und gemischt gegeneinander Testen

3. Verschiedene Layer Testen
