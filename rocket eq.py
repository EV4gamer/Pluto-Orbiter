import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


### constants ###
pluto_mass = 1.309 * 10**22 #kg
pluto_radius = 1188300      #m
G = 6.67 * 10 **-11
g0 = 9.80665


### orbit equations ###
def vis_viva(M, r, a):
    return np.sqrt(G * M * (2 / r - 1 / a))

def vcirc(M, r):
    return np.sqrt(G * M / r)

def vescape(M, r):
    return np.sqrt(2 * G * M / r)

def ecc(ra, rb):
    return (ra-rb) / (ra+rb)

def semimajor(ra, rb):
    return (ra + rb) / 2


### spacecraft equations ###
def massfrac(dv, isp):
    return np.exp(dv / (g0 * isp)) #returns total mass / (total mass - fuel)

def massrate(F, isp):
    return F / (g0 * isp * 1000) #tons / s

def fuelmass(dv, isp, w):
    ratio = massfrac(dv, isp)
    return ((ratio - 1) / (1 + (1 - ratio) * drymass_fraction)) * w

def craftmass(dv, isp, w):
    return w + drymass_fraction * fuelmass(dv, isp, w)

def totalmass(dv, isp, w):
    ratio = massfrac(dv, isp)
    return ((drymass_fraction + 1) * (ratio - 1) / (1 + (1 - ratio) * drymass_fraction) + 1) * w

def nucweight(PWe, factor=1):
    return (25.63 + PWe) / (0.0293 * 1000 * factor)

def plot(x, y):
    plt.plot(x, y, label=label[i], linestyle=lstyle[i], color = color[i])

def limitfx(valx, valy, lim = np.inf, limb = -np.inf):
    condition = np.where((valy > lim) | (valy < limb))[0]
    if len(condition) > 0:
        return valx[:condition[0]], valy[:condition[0]]
    else:
        return valx, valy
    
#current values for MPD assume a 200kW reactor
#VASMR can go from 1000 - 10000 s isp, but thrust level is not listed for all values
#sc weight does not include tank drymass
#NTO --> NTO + Aerozine 50

sc_weight = 0.4 #+ nucweight(200)                                     #ton
drymass_fraction = 0.1                                               #10%
engines = np.array([8, 5, 4, 1, 1, 1])                               #nr of engines

### chemical -- nuclear -- ion -- mpd ###
#th_weight = [0.1, 0.25, 2.2, 0.05, 2.6, 2.6]                        #ton
#isp = [320, 470, 950, 4170, 5000, 1000]                             #s
#thrust = [44000, 66000, 80000, 0.25, 6, 30]                         #N
#label = ['AJ10', 'RL10', 'BNTR', 'NEXT', 'MPD$^{[1]}$', 'MPD$^{[2]}$'] 

### ion -- mpd ###
th_weight = [0.05, 0.25, 0.4, 0.5, 2.6, 2.6] * engines               #ton
isp = [4170, 9600, 2900, 20000, 5000, 2500]                          #s
thrust = [0.25, 0.7, 2.3, 2.5, 6, 11] * engines                      #N
power = [7, 40, 50, 250, 200, 200] * engines   
label = ['NEXT', 'HiPEP', 'AEPS', 'DS4G', 'MPD$^{[1]}$', 'MPD$^{[2]}$']
mass = th_weight + sc_weight# + nucweight(power, 2)

lstyle = ['-', '-', '-', '-', '-', '--']
color = ['blue', 'orange', 'green', 'red', 'purple', 'purple']

delta_v = np.arange(5000, 17000, 1)


### normal plot dv vs mass###
for i in range(len(isp)):
    y = totalmass(delta_v, isp[i], mass[i])
    plot(*limitfx(delta_v, y, 100))    
plt.legend()
plt.ylabel('Mass (t)')
plt.xlabel('delta-v (ms$^{-1}$)')
plt.title('Plot of mass vs delta-v')
plt.show()


### normal plot burn time vs deltav ###
for i in range(len(isp)):
    y = fuelmass(delta_v, isp[i], mass[i]) / massrate(thrust[i], isp[i])
    plot(*limitfx(delta_v, y, 10**8, 0))    
plt.legend()
plt.ylabel('Burn time (s)')
plt.xlabel('delta-v (ms$^{-1}$)')
plt.title('Plot of burn time vs delta-v')
plt.show() 
