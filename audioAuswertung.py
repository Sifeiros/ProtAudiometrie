import numpy as np
import scipy as sp
import math as mt
import matplotlib.pyplot as plt


file_names = ["Audiometrie/audio_manuell.txt", "Audiometrie/audio_auto.txt"]


#Manuelles Audiogramm


man_audio_raw = file(file_names[0])

col_names = ["Frequenz [Hz]", "Prob. 1 L", "Prob. 1 R", "Prob. 2 L", "Prob. 2 R", "Prob. 3 L", "Prob. 3 R"]

man_audio = np.genfromtxt(man_audio_raw, names = col_names, dtype=float, skip_header=1)

print(man_audio[:,0])

plt.plot(man_audio[:][0], man_audio[:][1])

plt.show()