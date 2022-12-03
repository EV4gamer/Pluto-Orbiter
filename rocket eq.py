import numpy as np
import matplotlib.pyplot as plt

### constants ###
pluto_mass = 1.309 * 10**22 #kg
pluto_radius = 1188300      #m
G = 6.67 * 10 **-11
g0 = 9.81


### orbit equations ###
def vis_viva(M, r, a):
    return np.sqrt(G * M * (2 / r - 1 / a))

def vcirc(M, r):
    return np.sqrt(G * M / r)

def ecc(ra, rb):
    return (ra-rb) / (ra+rb)

def semimajor(ra, rb):
    return (ra + rb) / 2


### spacecraft equations ###
def massfrac(dv, isp):
    return np.exp(dv / (g0 * isp)) #returns total mass / (total mass - fuel)

def massrate(F, isp):
    return F / (g0 * isp * 1000) #tons / s

def totalmass(delta_v, isp, i):
    return massfrac(delta_v, isp) * (sc_weight + th_weight[i])

def fuelmass(delta_v, isp, i):
    return (totalmass(delta_v, isp, i) - (sc_weight + th_weight[i]))

def completemass(delta_v, isp, i):
    return totalmass(delta_v, isp, i) + fuelmass(delta_v, isp, i) * drymass_fraction

def plot(x, y):
    plt.plot(x, y, label=label[i], linestyle=lstyle[i], color = color[i])
    
#current values for MPD assume a 1MW reactor
#VASMR can go from 1500 - 10000 s isp, but thrust level is not listed for all values
#sc weight does not include tank drymass

#NTO --> NTO + UDMH + Hydrazine (NTO + Aerozine 50)
#Ion is NASA's NEXT thruster, using Xenon
sc_weight = 0.5                                                     #ton
drymass_fraction = 0.1                                              #10%
th_weight = [0.2, 0.25, 2.2, 0.05, 2.6, 2.6]                        #ton
isp = [320, 470, 950, 4200, 5000, 1000]                             #s
thrust = [88000, 66000, 80000, 0.25, 20, 100]                       #N
label = ['NTO-50', 'LH2', 'NTR-LH2', 'Ion', 'MPD$^{[1]}$', 'MPD$^{[2]}$'] 
lstyle = ['-', '-', '-', '-', '-', '--']
color = ['blue', 'orange', 'green', 'red', 'purple', 'purple']

delta_v = np.arange(5000, 15000, 1)

### log plot dv vs mass###
for i in range(len(isp)):
    plot(delta_v, np.log10(completemass(delta_v, isp[i], i)))
        
plt.legend()
plt.ylabel('Mass log$_{10}$(t)')
plt.xlabel('delta-v (ms$^{-1}$)')
plt.title('Logarithmic plot of mass vs delta-v')
plt.show()

### normal plot dv vs mass###
for i in range(len(isp)):
    plot(delta_v, completemass(delta_v, isp[i], i))
    
plt.legend()
plt.ylabel('Mass (t)')
plt.xlabel('delta-v (ms$^{-1}$)')
plt.title('Plot of mass vs delta-v')
plt.show()


### log plot dv vs tank mass###
# for i in range(len(isp)):
#     plot(delta_v, np.log10(completemass(delta_v, isp[i], i) - totalmass(delta_v, isp[i], i)))
        
# plt.legend()
# plt.ylabel('Tank mass log$_{10}$(t)')
# plt.xlabel('delta-v (ms$^{-1}$)')
# plt.title('Logarithmic plot of tank-mass vs delta-v')
# plt.show()

# ### normal plot dv vs tank mass###
# for i in range(len(isp)):
#     plot(delta_v, completemass(delta_v, isp[i], i) - totalmass(delta_v, isp[i], i))
    
# plt.legend()
# plt.ylabel('Tank mass (t)')
# plt.xlabel('delta-v (ms$^{-1}$)')
# plt.title('Plot of tank-mass vs delta-v')
# plt.show()

### log plot dv vs tank mass###
# for i in range(len(isp)):
#     plot(delta_v, np.log10(fuelmass(delta_v, isp[i], i)))
        
# plt.legend()
# plt.ylabel('Fuel mass log$_{10}$(t)')
# plt.xlabel('delta-v (ms$^{-1}$)')
# plt.title('Logarithmic plot of fuel-mass vs delta-v')
# plt.show()

# ### normal plot dv vs tank mass###
# for i in range(len(isp)):
#     plot(delta_v, fuelmass(delta_v, isp[i], i))
    
# plt.legend()
# plt.ylabel('Fuel mass (t)')
# plt.xlabel('delta-v (ms$^{-1}$)')
# plt.title('Plot of fuel-mass vs delta-v')
# plt.show()


### log plot burn time vs deltav ###
for i in range(len(isp)):
    plot(delta_v, np.log10(fuelmass(delta_v, isp[i], i) / massrate(thrust[i], isp[i])))

plt.legend()
plt.ylabel('Burn time log$_{10}$(s)')
plt.xlabel('delta-v (ms$^{-1}$)')
plt.title('Logarithmic plot of burn time vs delta-v')
plt.show() 

### normal plot burn time vs deltav ###
for i in range(len(isp)):
    plot(delta_v, fuelmass(delta_v, isp[i], i) / massrate(thrust[i], isp[i]))
    
plt.legend()
plt.ylabel('Burn time (s)')
plt.xlabel('delta-v (ms$^{-1}$)')
plt.title('Plot of burn time vs delta-v')
plt.show() 
