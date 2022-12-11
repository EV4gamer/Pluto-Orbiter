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

def fuelmass(dv, isp, thrust_w):
    ratio = massfrac(dv, isp)
    return ((ratio - 1) / (1 + (1 - ratio) * drymass_fraction)) * (sc_weight + thrust_w)

def craftmass(dv, isp, thrust_w):
    return sc_weight + thrust_w + drymass_fraction * fuelmass(dv, isp, thrust_w)

def totalmass(dv, isp, thrust_w):
    ratio = massfrac(dv, isp)
    return ((drymass_fraction + 1) * (ratio - 1) / (1 + (1 - ratio) * drymass_fraction) + 1) * (sc_weight + thrust_w)



def plot(x, y):
    plt.plot(x, y, label=label[i], linestyle=lstyle[i], color = color[i])

def limitfx(valx, valy, lim, i, i_lim):
    if i < i_lim:
        return valx[:np.where(valy > lim)[0][0]], valy[:np.where(valy > lim)[0][0]]
    else:
        return valx, valy
    
#current values for MPD assume a 200kW reactor
#VASMR can go from 1000 - 10000 s isp, but thrust level is not listed for all values
#sc weight does not include tank drymass
#NTO --> NTO + Aerozine 50
#Ion is NASA's NEXT thruster, using Xenon

sc_weight = 0.4                                                     #ton
drymass_fraction = 0.1                                              #10%
th_weight = [0.2, 0.25, 2.2, 0.05, 2.6, 2.6]                        #ton
isp = [320, 470, 950, 4200, 5000, 1000]                             #s
thrust = [88000, 66000, 80000, 0.25, 6, 30]                         #N
label = ['NTO-50', 'LH2', 'NTR-LH2', 'Ion', 'MPD$^{[1]}$', 'MPD$^{[2]}$'] 
lstyle = ['-', '-', '-', '-', '-', '--']
color = ['blue', 'orange', 'green', 'red', 'purple', 'purple']

delta_v = np.arange(3000, 15000, 1)

### log plot dv vs mass###
for i in range(len(isp)):
    plot(delta_v, np.log10(totalmass(delta_v, isp[i], th_weight[i])))
        
plt.legend()
plt.ylabel('Mass log$_{10}$(t)')
plt.xlabel('delta-v (ms$^{-1}$)')
plt.title('Logarithmic plot of mass vs delta-v')
plt.show()


### normal plot dv vs mass###
for i in range(len(isp)):
    y = totalmass(delta_v, isp[i], th_weight[i])
    plot(*limitfx(delta_v, y, 100, i, 2))
    
plt.legend()
plt.ylabel('Mass (t)')
plt.xlabel('delta-v (ms$^{-1}$)')
plt.title('Plot of mass vs delta-v')
plt.show()


### log plot burn time vs deltav ###
for i in range(len(isp)):
    plot(delta_v, np.log10(fuelmass(delta_v, isp[i], th_weight[i]) / massrate(thrust[i], isp[i])))

plt.legend()
plt.ylabel('Burn time log$_{10}$(s)')
plt.xlabel('delta-v (ms$^{-1}$)')
plt.title('Logarithmic plot of burn time vs delta-v')
plt.show() 

### normal plot burn time vs deltav ###
for i in range(len(isp)):
    y = fuelmass(delta_v, isp[i], th_weight[i]) / massrate(thrust[i], isp[i])
    plot(*limitfx(delta_v, y, 10**4, i, 2))
    
plt.legend()
plt.ylabel('Burn time (s)')
plt.xlabel('delta-v (ms$^{-1}$)')
plt.title('Plot of burn time vs delta-v')
plt.show() 


### plot dv vs mass per dv###
for i in range(len(isp)):
    y = totalmass(delta_v, isp[i], th_weight[i]) / delta_v
    plot(*limitfx(delta_v, y, 0.0035, i, 2))
    
plt.legend()
plt.ylabel('mass / delta-v (t (ms$^{-1}$)$^{-1}$)')
plt.xlabel('delta-v (ms$^{-1}$)')
plt.title('Plot of mass / delta-v vs delta-v')
plt.show()
