# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import math as mt
import matplotlib.pyplot as plt
import scipy.optimize as optimization


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

    ax1[i/2].set_xscale('log')
    
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

    ax2[i/2].set_xscale('log')

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


detail_audio1_raw = file(file_names[2])
detail_audio2_raw = file(file_names[3])

col_names1 = ["Frequenz [Hz]", "L", "R"]
col_names2 = ["Frequenz [Hz]", "L 15 dB", "L 30 dB"]


colors1 = ["blue", "red"]
colors2 = ["blue", "blue", "blue"]
markers1 = ["o", "^"]
markers2 = ["o", "^", "x"]
titles = ["Proband 2: detailliertes Audiogramm", u"Linkes Ohr mit versch. Dämpfungen"]


detail_audio1 = np.genfromtxt(detail_audio1_raw, dtype=float, skip_header=1)
detail_audio1 = detail_audio1.transpose()
detail_audio1[1] = 0.5*detail_audio1[1]
detail_audio1[2] = 0.5*detail_audio1[2]
detail_audio2 = np.genfromtxt(detail_audio2_raw, dtype=float, skip_header=1)
detail_audio2 = detail_audio2.transpose()
detail_audio2[1] = 0.5*detail_audio2[1]
detail_audio2[2] = 0.5*detail_audio2[2]




xticks = []
ind = []
for i in range(len(detail_audio1[0][::3])):
    xticks.append("")
    print(detail_audio1[0][3*i])
    ind.append(3*i)
    


print(detail_audio1[0][::3])
ticks = ["80", "375", "1k", "2k", "3,5k", "6k", "10k", "16k" ]
ind = [0,3,6,9,11,13,15,17]

#print(detail_audio1[0][ind])

#print(len(ind))
#print(len(ticks))
#print(len(xticks))

for in1 in range(len(ind)):
    xticks[ind[in1]] = ticks[in1]
    print(ticks[in1])
    #print(ticks[in1])


"""
fig3, ax3 = plt.subplots(2, sharex=True)

ax3[0].set_title(titles[0])
ax3[0].plot(detail_audio1[0], detail_audio1[2], label=col_names1[2], color=colors1[0], marker=markers1[0])
ax3[0].plot(detail_audio1[0], detail_audio1[1], label=col_names1[1], color=colors1[1], marker=markers1[1])

ax3[0].set_xscale('log')

ax3[0].get_xaxis().set_ticks(detail_audio1[0][::4])
ax3[0].get_xaxis().set_ticklabels(xticks)
#ax3.set_xbound()
#ax3.set_ybound()
ax3[0].legend(loc=0)
ax3[0].set_ylabel(u"Hörschwelle [dB-HL]")
ax3[0].grid(b=True, which="major")


ax3[1].set_title(titles[1])
ax3[1].plot(detail_audio1[0], detail_audio1[1], label=col_names1[1], color=colors2[0], marker=markers2[0])
ax3[1].plot(detail_audio2[0], detail_audio2[1], label=col_names2[1], color=colors2[1], marker=markers2[1])
ax3[1].plot(detail_audio2[0], detail_audio2[2], label=col_names2[2], color=colors2[2], marker=markers2[2])

ax3[1].set_xscale('log')

ax3[1].get_xaxis().set_ticks(detail_audio1[0][::4])
ax3[1].get_xaxis().set_ticklabels(xticks)
ax3[1].legend(loc=0)    
ax3[1].set_xlabel(col_names1[0])
ax3[1].set_ylabel(u"Hörschwelle [dB-HL]")
ax3[1].grid(b=True, which="major")
"""



"""
fig4, ax4 = plt.subplots(2, sharex=True)

ax4[0].set_title(titles[0])
ax4[0].plot(detail_audio1[0], detail_audio1[2], label=col_names1[2], color=colors1[0], marker=markers1[0])
ax4[0].plot(detail_audio1[0], detail_audio1[1], label=col_names1[1], color=colors1[1], marker=markers1[1])

ax4[0].set_xscale('log')

ax4[0].get_xaxis().set_ticks(detail_audio1[0][::4])
ax4[0].get_xaxis().set_ticklabels(xticks)
ax4[0].set_xbound(lower=0, upper=9000)
ax4[0].legend(loc=0)
ax4[0].set_ylabel(u"Hörschwelle [dB-V]")
ax4[0].grid(b=True, which="major")



ax4[1].set_title(titles[1])
ax4[1].plot(detail_audio1[0], detail_audio1[1], label=col_names1[1], color=colors2[0], marker=markers2[0])
ax4[1].plot(detail_audio2[0], detail_audio2[1], label=col_names2[1], color=colors2[1], marker=markers2[1])
ax4[1].plot(detail_audio2[0], detail_audio2[2], label=col_names2[2], color=colors2[2], marker=markers2[2])

ax4[1].set_xscale('log')

ax4[1].get_xaxis().set_ticks(detail_audio1[0][::4])
ax4[1].get_xaxis().set_ticklabels(xticks)
ax4[1].set_xbound(lower=0, upper=9000)
ax4[1].legend(loc=0)
ax4[1].set_ylabel(u"Hörschwelle [dB-V]")
ax4[1].grid(b=True, which="major")

ax4[1].set_xlabel(col_names1[0])
"""

fig4, ax4 = plt.subplots(3, sharex=True)


ax4[0].set_title(titles[0])
ax4[0].plot(detail_audio1[0], detail_audio1[2], label=col_names1[2], color=colors1[0], marker=markers1[0])
ax4[0].plot(detail_audio1[0], detail_audio1[1], label=col_names1[1], color=colors1[1], marker=markers1[1])

ax4[0].set_xscale('log')

ax4[0].get_xaxis().set_ticks(detail_audio1[0][::3])
ax4[0].get_xaxis().set_ticklabels(xticks)
ax4[0].set_xbound(lower=0, upper=9000)
ax4[0].legend(loc=0)
ax4[0].set_ylabel(u"Hörschwelle [dB-V]")
ax4[0].grid(b=True, which="major")



ax4[1].set_title(titles[1])
ax4[1].plot(detail_audio1[0], detail_audio1[1], label=col_names1[1], color=colors2[0], marker=markers2[0])
ax4[1].plot(detail_audio2[0], detail_audio2[1], label=col_names2[1], color=colors2[1], marker=markers2[1])
ax4[1].plot(detail_audio2[0], detail_audio2[2], label=col_names2[2], color=colors2[2], marker=markers2[2])

ax4[1].set_xscale('log')

ax4[1].get_xaxis().set_ticks(detail_audio1[0][::3])
ax4[1].get_xaxis().set_ticklabels(xticks)
ax4[1].set_xbound(lower=0, upper=9000)
ax4[1].legend(loc=0)
ax4[1].set_ylabel(u"Hörschwelle [dB-V]")
ax4[1].grid(b=True, which="major")

ax4[1].set_xlabel(col_names1[0])



"""
Conversion from db SPL to dB HL taken from http://ec.europa.eu/health/ph_risk/committees/04_scenihr/docs/scenihr_o_018.pdf 

Table at p.18
Graph at p.17
ISO 226:2003

Otherwise conversion via ANSI Norm

"""
#dB_conversion_ISO = [[31.5,63,125,250,500,1000,2000,4000,8000],
#                 [60,38,22,12,5,2,-2,-5,13]]


dB_conversion_ISO = np.array([[20,40,80,100,200,400,500,800,1000,1400,2000,3000,4000,8000,10000,14000,17001],
                              [75,50,30,25,14,6,5,3,2,4,-2,-7,-5,13,16,13,30]])

"""
v0 = np.array([1.,1.,1.,1.,1.,1.])
sigma = 0.001*np.ones(17, dtype="float")
print(sigma)


def model(x, a, b, c, d, e, f):
    return a + b*np.log(x) + c*np.log(x)**2. + d*np.log(x)**3. + e*np.log(x)**4. + f*np.log(x)**5.

fit_param = optimization.curve_fit(model, dB_conversion_ISO[0], dB_conversion_ISO[1], v0, sigma)[0]

def model2(x):
    return fit_param[0] + fit_param[1]*np.log(x) + fit_param[2]*np.log(x)**2. + fit_param[3]*np.log(x)**3. + fit_param[4]*np.log(x)**4. + fit_param[5]*np.log(x)**5.

#dB_conversion_ISO[0] = (dB_conversion_ISO[0])

#dB_conversion_ISO = np.array(dB_conversion)

#dB_conversion_ANSI = [[40,50,60,80,100,200,400,500,800,1000,2000,4000,5000,8000,10000],[]]

"""
#------------------------------------------------------------------
#Interpolation for HL to SPS conversion
#------------------------------------------------------------------
def log_interpol(fun_array, x):
    #needs sorted base array fun_array

    #print("x = " + str(x))
    for i in range(len(fun_array[0])):
        #print(fun_array[0][i])
        if x < fun_array[0][i]:
            x0 = fun_array[0][i-1]
            x1 = fun_array[0][i]
            y0 = fun_array[1][i-1]
            y1 = fun_array[1][i]
            break
    #Log Interpolation between data points:
    #y0 = m*ln(x0) + n
    m = (y0-y1)/(np.log(x0)-np.log(x1))
    n = y0-m*np.log(x0)

    return m*np.log(x) + n

#test = range(100, 15000, 100)
#val = []

#print(dB_conversion_ISO)

#for el in test:
#    val.append(log_interpol(dB_conversion_ISO, el))

#dat_plot = np.array([test,val])


fig5, ax5 = plt.subplots(1)
ax5.set_title("Isophonie-Kurve nach ISO")
ax5.plot(dB_conversion_ISO[0], dB_conversion_ISO[1], marker = "^", color="blue")
#ax5.plot(dat_plot[0], dat_plot[1], marker = "o", color= "red")
ax5.set_xscale('log')
ax5.grid(b=True)
ax5.get_xaxis().set_ticks(detail_audio1[0][::3])
ax5.get_xaxis().set_ticklabels(xticks)
ax5.set_xlabel("Frequenz [Hz]")
ax5.set_ylabel(u"Hörschwelle [dB-SPL]")

#ax5.get_xaxis().set_ticks(detail_audio1[0][::4])
#ax5.get_xaxis().set_ticklabels(xticks)



#---------------------------------------------
# Assumption of constant relation between Intensity of electric signale and acoustic response
# To determine offset, compare value at 1 kHz from detailed audiogram and manual audiogramm with medical equipment
# The left ear is chosen, because the right headphone seemed to have some problems
#---------------------------------------------


db_SPL_offset = +34. #dB, conversion dB-V to dB-SPL

detail_audio1_spl = detail_audio1.copy()
detail_audio2_spl = detail_audio2.copy()

detail_audio1_spl[1] = detail_audio1_spl[1] + db_SPL_offset
detail_audio2_spl[1] = detail_audio2_spl[1] + db_SPL_offset
detail_audio1_spl[2] = detail_audio1_spl[2] + db_SPL_offset
detail_audio2_spl[2] = detail_audio2_spl[2] + db_SPL_offset


fig6, ax6 = plt.subplots(2, sharex=True)

ax6[0].set_title(titles[0])
ax6[0].plot(detail_audio1_spl[0], detail_audio1_spl[2], label=col_names1[2], color=colors1[0], marker=markers1[0])
ax6[0].plot(detail_audio1_spl[0], detail_audio1_spl[1], label=col_names1[1], color=colors1[1], marker=markers1[1])

ax6[0].set_xscale('log')

ax6[0].get_xaxis().set_ticks(detail_audio1[0][::3])
ax6[0].get_xaxis().set_ticklabels(xticks)
ax6[0].set_xbound(lower=0, upper=9000)
ax6[0].legend(loc=0)
ax6[0].set_ylabel(u"Hörschwelle [dB-SPL]")
ax6[0].grid(b=True, which="major")



ax6[1].set_title(titles[1])
ax6[1].plot(detail_audio1_spl[0], detail_audio1_spl[1], label=col_names1[1], color=colors2[0], marker=markers2[0])
ax6[1].plot(detail_audio2_spl[0], detail_audio2_spl[1], label=col_names2[1], color=colors2[1], marker=markers2[1])
ax6[1].plot(detail_audio2_spl[0], detail_audio2_spl[2], label=col_names2[2], color=colors2[2], marker=markers2[2])

ax6[1].set_xscale('log')

ax6[1].get_xaxis().set_ticks(detail_audio1_spl[0][::3])
ax6[1].get_xaxis().set_ticklabels(xticks)
ax6[1].set_xbound(lower=0, upper=17000)
ax6[1].legend(loc=0)
ax6[1].set_ylabel(u"Hörschwelle [dB-SPL]")
ax6[1].grid(b=True, which="major")

ax6[1].set_xlabel(col_names1[0])


#---------------------------------------------------------
#Correction for human hearing level
#---------------------------------------------------------

detail_audio1_hl = detail_audio1_spl.copy()
detail_audio2_hl = detail_audio2_spl.copy()

for i in range(len(detail_audio1_hl[0])):
    detail_audio1_hl[1][i] += log_interpol(dB_conversion_ISO, detail_audio1_hl[0][i])
    detail_audio1_hl[2][i] += log_interpol(dB_conversion_ISO, detail_audio1_hl[0][i])


for i in range(len(detail_audio2_hl)):
    detail_audio2_hl[1][i] += log_interpol(dB_conversion_ISO, detail_audio2_hl[0][i])
    detail_audio2_hl[2][i] += log_interpol(dB_conversion_ISO, detail_audio2_hl[0][i])



fig7, ax7 = plt.subplots(2, sharex=True)

ax7[0].set_title(titles[0])
ax7[0].plot(detail_audio1_hl[0], detail_audio1_hl[2], label=col_names1[2], color=colors1[0], marker=markers1[0])
ax7[0].plot(detail_audio1_hl[0], detail_audio1_hl[1], label=col_names1[1], color=colors1[1], marker=markers1[1])

ax7[0].set_xscale('log')

ax7[0].get_xaxis().set_ticks(detail_audio1[0][::3])
ax7[0].get_xaxis().set_ticklabels(xticks)
ax7[0].set_xbound(lower=0, upper=17000)
ax7[0].legend(loc=0)
ax7[0].set_ylabel(u"Hörschwelle [dB-HL]")
ax7[0].grid(b=True, which="major")



ax7[1].set_title(titles[1])
ax7[1].plot(detail_audio1_hl[0], detail_audio1_hl[1], label=col_names1[1], color=colors2[0], marker=markers2[0])
ax7[1].plot(detail_audio2_hl[0], detail_audio2_hl[1], label=col_names2[1], color=colors2[1], marker=markers2[1])
ax7[1].plot(detail_audio2_hl[0], detail_audio2_hl[2], label=col_names2[2], color=colors2[2], marker=markers2[2])

ax7[1].set_xscale('log')

ax7[1].get_xaxis().set_ticks(detail_audio1_spl[0][::3])
ax7[1].get_xaxis().set_ticklabels(xticks)
ax7[1].set_xbound(lower=0, upper=17000)
ax7[1].legend(loc=0)
ax7[1].set_ylabel(u"Hörschwelle [dB-HL]")
ax7[1].grid(b=True, which="major")

ax7[1].set_xlabel(col_names1[0])
#---------------------------------------------------------


fig8, ax8 = plt.subplots(3, sharex=True)


ax8[0].set_title(titles[0])
ax8[0].plot(detail_audio1[0], detail_audio1[2], label=col_names1[2], color=colors1[1], marker=markers1[0])
ax8[0].plot(detail_audio1[0], detail_audio1[1], label=col_names1[1], color=colors1[0], marker=markers1[1])

ax8[0].set_xscale('log')
ax8[0].get_xaxis().set_ticks(detail_audio1[0][::3])
ax8[0].get_xaxis().set_ticklabels(xticks)
ax8[0].set_xbound(lower=0, upper=17000)
ax8[0].legend(loc=0)
ax8[0].set_ylabel(u"Schwelle[dB-V]")
ax8[0].grid(b=True, which="major")



#ax8[1].set_title(titles[1])
ax8[1].plot(detail_audio1_spl[0], detail_audio1_spl[2], label=col_names1[2], color=colors1[1], marker=markers1[0])
ax8[1].plot(detail_audio1_spl[0], detail_audio1_spl[1], label=col_names1[1], color=colors1[0], marker=markers1[1])

ax8[1].set_xscale('log')
ax8[1].get_xaxis().set_ticks(detail_audio1[0][::3])
ax8[1].get_xaxis().set_ticklabels(xticks)
ax8[1].set_xbound(lower=0, upper=17000)
ax8[1].legend(loc=0)
ax8[1].set_ylabel(u"Schwelle[dB-SPL]")
ax8[1].grid(b=True, which="major")




#ax8[2].set_title(titles[1])
ax8[2].plot(detail_audio1_hl[0], detail_audio1_hl[2], label=col_names1[2], color=colors1[1], marker=markers1[0])
ax8[2].plot(detail_audio1_hl[0], detail_audio1_hl[1], label=col_names1[1], color=colors1[0], marker=markers1[1])

ax8[2].set_xscale('log')
ax8[2].get_xaxis().set_ticks(detail_audio1[0][::3])
ax8[2].get_xaxis().set_ticklabels(xticks)
ax8[2].set_xbound(lower=0, upper=17000)
ax8[2].legend(loc=0)
ax8[2].set_ylabel(u"Schwelle[dB-HL]")
ax8[2].grid(b=True, which="major")


ax8[2].set_xlabel(col_names1[0])



#---------------------------------------------------------
plt.show()

