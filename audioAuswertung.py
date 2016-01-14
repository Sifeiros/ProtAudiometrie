# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import math as mt
import matplotlib.pyplot as plt

direc = "Daten/"
file_names = [direc+"audio_manuell.txt", direc+"audio_auto.txt", direc+"detail_audio1.txt", direc+"detail_audio2.txt"]


#Manuelles Audiogramm
#------------------------------------------

man_audio_raw = file(file_names[0])
col_names = ["Frequenz [Hz]", "L", "R", "L", "R", "L", "R"]

xticks = ["125","","500","","1k","","2k","","4k","","8k"]
colors = ["blue", "blue", "red", "red", "black", "black"]
markers = ["o", "^", "o", "^", "o", "^"]
titles = ["Proband 1", "Proband 2", "Proband 3"]

man_audio = np.genfromtxt(man_audio_raw, dtype=float, skip_header=1)
#print(type(man_audio))
man_audio = man_audio.transpose()



fig1, ax1 = plt.subplots(3, sharex=True)

for i in range(0,len(col_names)-1):
    ax1[i/2].set_title(titles[i/2])
    ax1[i/2].plot(man_audio[0], man_audio[i+1], label=col_names[i+1], color=colors[i], marker=markers[i])
    ax1[i/2].grid(b=True, which="major")
    ax1[i/2].get_xaxis().set_ticks(man_audio[0])
    ax1[i/2].get_xaxis().set_ticklabels(xticks)
    ax1[i/2].set_xbound(lower=0, upper=man_audio[0][-1]+500)
    ax1[i/2].set_ybound(lower=-15, upper=25)
    ax1[i/2].legend(loc=0)
    
ax1[1].set_ylabel(u"Hörschwelle [dB-HL]")
plt.xlabel("Frequenz [Hz]")


#Automatisches Audiogramm mit Knochenleitung
#--------------------------------------------------------

aut_audio_raw = file(file_names[1])
col_names = ["Frequenz [Hz]", "R", "L", "R", "L"]

xticks = ["500", "", "2k", "", "4k", "", "8k"]
colors = ["red", "blue", "red", "blue"]
markers = ["o", "^", "o", "^"]
titles = ["Luftleitung", "Knochenleitung"]

aut_audio = np.genfromtxt(aut_audio_raw, dtype=float, skip_header=1)
aut_audio=aut_audio.transpose()



fig2, ax2 = plt.subplots(2, sharex=True)


for i in range(0, len(col_names)-1):
    ax2[i/2].set_title(titles[i/2])
    ax2[i/2].plot(aut_audio[0], aut_audio[i+1], label=col_names[i+1], color=colors[i], marker=markers[i])
    ax2[i/2].grid(b=True, which="major")
    ax2[i/2].get_xaxis().set_ticks(aut_audio[0])
    ax2[i/2].get_xaxis().set_ticklabels(xticks)
    ax2[i/2].set_xbound(lower=0, upper=aut_audio[0][-1]+500)
    ax2[i/2].set_ybound(lower=-15, upper=25)
    ax2[i/2].legend(loc=0)
    ax2[i/2].set_ylabel(u"Hörschwelle [dB-HL]")
    

plt.xlabel("Frequenz [Hz]")


#Detailliertes Audiogramm
#-------------------------------------------------------

detail_audio1 = file(file_names[2])
detail_audio2 = file(file_names[3])




plt.show()