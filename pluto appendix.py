import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from copy import deepcopy

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


sc_weight = 1.5                                                      #ton
drymass_fraction = 0.1                                               #10%
delta_v = np.arange(5000, 17000, 1)

### spacecraft equations ###
def massfrac(dv, isp):
    return np.exp(dv / (g0 * isp)) #returns total mass / (total mass - fuel)

def massrate(F, isp):
    return F / (g0 * isp * 1000) #tons / s

def fuelmass(dv, isp, w, df=drymass_fraction):
    ratio = massfrac(dv, isp)
    return ((ratio - 1) / (1 + (1 - ratio) * df)) * w

def craftmass(dv, isp, w, df=drymass_fraction):
    return w + df * fuelmass(dv, isp, w)

def totalmass(dv, isp, w, df=drymass_fraction):
    ratio = massfrac(dv, isp)
    return ((df + 1) * (ratio - 1) / (1 + (1 - ratio) * df) + 1) * w

def burntime(dv, engine): #burntime in days
    if engine.label == 'AJ10' or engine.label == 'RL10':
        return nstage(engine, dv) / engine.massrate() / (24 * 3600)
    else:
        return engine.burntime() / (24 * 3600)

def nucweight(PWe, factor=1):
    if PWe == 0:
        return 0
    else:
        return (25.63 + PWe) / (0.0293 * 1000 * factor)
        #return PWe / 50
        
def vasimr_weight(PWe, factor=1): #ton
    return (PWe * 1.2 + 444) / (1000 * factor)

def vt_ar(PWe):
    return (PWe / 200) * 6

def vt_kr(PWe):
    return (PWe / 200) * 11

def plot(x, y):
    plt.plot(x, y, label=engine.label)

def limitfx(valx, valy, lim = np.inf, limb = -np.inf):
    condition = np.where((valy > lim) | (valy < limb))[0]
    if len(condition) > 0:
        return valx[:condition[0]], valy[:condition[0]]
    else:
        return valx, valy

def nstage(engine, dv):
    AJdv1 = 6000
    AJdv2 = 11500
    RLdv = 8000
    
    v = []
    #atm only aj10, rl10
    for index in dv:
        if engine.label == 'AJ10':
            if index > AJdv1 and index <= AJdv2:
                v += [fuelmass(index - AJdv1, engine.isp, totalmass(AJdv1, engine.isp, engine.mass()) + engine.weight) + fuelmass(AJdv1, engine.isp, engine.mass())]
            elif index > AJdv2:
                tfuelmass = fuelmass(AJdv2 - AJdv1, engine.isp, totalmass(AJdv1, engine.isp, engine.mass()) + engine.weight) + fuelmass(AJdv1, engine.isp, engine.mass())
                tmass = (1 + drymass_fraction) * tfuelmass + sc_weight + engine.weight * 2
                v += [tfuelmass + fuelmass(index - AJdv2, engine.isp, tmass)]
            else:
                v += [fuelmass(index, engine.isp, engine.mass())]
        if engine.label == 'RL10':
            if index > RLdv:
                v += [fuelmass(index - RLdv, engine.isp, totalmass(RLdv, engine.isp, engine.mass()) + engine.weight) + fuelmass(RLdv, engine.isp, engine.mass())]
            else:
                v += [fuelmass(index, engine.isp, engine.mass())] 
    return np.array(v)


class engine:
    def __init__(self, weight, isp, thrust, power, engines, label, tf=drymass_fraction):
        self.weight = weight
        self.isp = isp
        self.thrust = thrust
        self.power = power
        self.label = label
        self.engines = engines
        self.drymass_fraction = tf
        
    def mass(self): #engines + s/c mass + nuc
        return self.weight * self.engines + sc_weight + nucweight(self.power * self.engines, 2)
    
    def massrate(self): #tons / s
        return self.thrust * self.engines / (g0 * self.isp * 1000)
    
    def fuelmass(self, dv):
        ratio = massfrac(dv, self.isp)
        return ((ratio - 1) / (1 + (1 - ratio) * self.drymass_fraction)) * self.mass()   
    
    def totalmass(self, dv):
        return (1 + self.drymass_fraction) * self.fuelmass(dv) + self.mass()
    
    def burntime(self, dv, days=False):
        if days:
            return self.fuelmass(dv) / self.massrate() / (24 * 3600)
        else:    
            return self.fuelmass(dv) / self.massrate()
    
    def properties(self, dv):
        return self.fuelmass(dv), self.totalmass(dv), self.burntime(dv) / (24 * 3600)
    
engine_list = [
    #engine(weight, isp,    thrust, power,  engines, label,     tf) 
    engine(0.1,                         320,    44000,      0,      1,  'AJ10',         0.10),
    engine(0.25,                        470,    66000,      0,      1,  'RL10',         0.10),
    engine(2.2,                         950,    80000,      0,      1,  'BNTR',         0.10),
    engine(0.05,                        4170,   0.237,      7,      8,  'NEXT',         0.09),
    engine(0.25,                        9620,   0.67,       39.3,   4,  'HiPEP',        0.09),
    engine(0.5,                         2900,   1.7,        50,     2,  'AEPS',         0.09),
    engine(0.8,                         19300,  2.5,        250,    1,  'DS4G',         0.09),
    engine(0.6,                         2650,   5.4,        100,    1,  'X3',           0.09),
    engine(vasimr_weight(120, 1.1),     5000,   vt_ar(120), 120,    1,  'MPD$_{Ar}$',   0.05),
    engine(vasimr_weight(120, 1.1),     2500,   vt_ar(120), 120,    1,  'MPD$_{Kr}$',   0.05),
    engine(2.8,                         10141,  20.1,       0,      1,  'DFD') #uses 2000kWe, but weight includes reactor
    ]


### log plot dv vs mass###
for engine in engine_list:
    if(engine.label == 'AJ10' or engine.label == 'RL10'):
        plot(delta_v, nstage(engine, delta_v) * (1 + engine.drymass_fraction) + engine.mass())
    else:
        plot(delta_v, engine.totalmass(delta_v))  
plt.grid()
plt.legend()
plt.gca().set_yscale('log')
plt.ylabel('Mass (t)')
plt.xlabel('delta-v (ms$^{-1}$)')
plt.title('Logarithmic plot of mass vs delta-v')
plt.tight_layout()
plt.show()

### log plot burn time vs deltav ###
for engine in engine_list:
    if(engine.label == 'AJ10' or engine.label == 'RL10'):
        plot(delta_v, nstage(engine, delta_v) / engine.massrate())
    else:
        plot(delta_v, engine.burntime(delta_v))
plt.grid()
plt.tight_layout()
plt.legend()
plt.gca().set_yscale('log')
plt.ylabel('Burn time (s)')
plt.xlabel('delta-v (ms$^{-1}$)')
plt.title('Logarithmic plot of burn time vs delta-v')
plt.show() 


### plot dv vs mass per dv###
for engine in engine_list:
    if(engine.label == 'AJ10' or engine.label == 'RL10'):
        y = (nstage(engine, delta_v) + engine.mass()) / delta_v
    else:
        y = engine.totalmass(delta_v) / delta_v        
    plot(*limitfx(delta_v, y, .01, 0))    
plt.grid()
plt.tight_layout()
plt.legend()
plt.ylabel('mass / delta-v (t (ms$^{-1}$)$^{-1}$)')
plt.xlabel('delta-v (ms$^{-1}$)')
plt.title('Plot of mass / delta-v vs delta-v')
plt.show()

x0, xl = 4, 11
y0, yl = 100, 600
dv = 17000
for engine in deepcopy(engine_list): #copy by value, not reference
    if (engine.label != "MPD$_{Ar}$") and (engine.label != "MPD$_{Kr}$"):
        burn = []
        tmass = []
        for i in range(1, 5):
            engine.engines = i
            burn += [engine.burntime(dv, True)]
            tmass += [engine.totalmass(dv)]
            if(tmass[-1] < xl) and (tmass[-1] > 0) and (burn[-1] < yl) and (burn[-1] > 0):
                plt.annotate(str(i), (tmass[-1], burn[-1]))
        plt.plot(tmass, burn, label=engine.label)
   
plt.tight_layout()
plt.xlabel('Total s/c mass (t)')
plt.ylabel('Burntime (days)')
plt.title("Total mass vs burntime per engine or per kWe\nat "+str(dv)+' kms$^{-1}$')


tmass = []
burn = []
for p in np.arange(50, 150):
    vmass = vasimr_weight(p, 1.1)
    nmass = nucweight(p, 2)
    tmass += [totalmass(dv, 5000, vmass + sc_weight + nmass, 0.05)]
    burn += [fuelmass(dv, 5000, vmass + sc_weight + nmass, 0.05) / massrate(6 * p / 200, 5000) / (24 * 3600)]
plt.plot(tmass, burn, label='MPD$_{Ar}$')
plt.annotate('50kw', (tmass[0], burn[0]))
plt.plot(tmass[0], burn[0], '.', color='black')
plt.annotate('150kw', (tmass[-1], burn[-1]))
plt.plot(tmass[-1], burn[-1], '.', color='black')



tmass = []
burn = []
for p in np.arange(50, 150):
    vmass = vasimr_weight(p, 1.1)
    nmass = nucweight(p, 2)
    tmass += [totalmass(dv, 2500, vmass + sc_weight + nmass, 0.05)]
    burn += [fuelmass(dv, 2500, vmass + sc_weight + nmass, 0.05) / massrate(11 * p / 200, 2500) / (24 * 3600)]
plt.plot(tmass, burn, label='MPD$_{Kr}$')
plt.annotate('50kw', (tmass[0], burn[0]))
plt.plot(tmass[0], burn[0], '.', color='black')
plt.annotate('150kw', (tmass[-1], burn[-1]))
plt.plot(tmass[-1], burn[-1], '.', color='black')


plt.ylim(y0, yl)
plt.xlim(x0, xl)
plt.legend()
plt.tight_layout()
plt.show()
