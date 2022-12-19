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

def burntime(dv, engine): #burntime in days
    if engine.label == 'AJ10' or engine.label == 'RL10':
        return nstage(engine, dv) / engine.massrate() / (24 * 3600)
    else:
        return fuelmass(dv, engine.isp, engine.mass()) / engine.massrate() / (24 * 3600)

def nucweight(PWe, factor=1):
    if PWe == 0:
        return 0
    else:
        return (25.63 + PWe) / (0.0293 * 1000 * factor)

def vasimr_weight(PWe):
    return PWe * 1.2 + 444

def plot(x, y):
    plt.plot(x, y, label=engine.label)

def limitfx(valx, valy, lim = np.inf, limb = -np.inf):
    condition = np.where((valy > lim) | (valy < limb))[0]
    if len(condition) > 0:
        return valx[:condition[0]], valy[:condition[0]]
    else:
        return valx, valy

def nstage(engine, dv):
    v = []
    #atm only aj10, rl10
    for index in dv:
        if engine.label == 'AJ10':
            if index > 7000 and index <= 11500:
                v += [fuelmass(index - 7000, engine.isp, totalmass(7000, engine.isp, engine.mass()) + engine.weight) + fuelmass(7000, engine.isp, engine.mass())]
            elif index > 11500:
                tfuelmass = fuelmass(11500 - 7000, engine.isp, totalmass(7000, engine.isp, engine.mass()) + engine.weight) + fuelmass(7000, engine.isp, engine.mass())
                tmass = (1 + drymass_fraction) * tfuelmass + sc_weight + engine.weight * 2
                v += [tfuelmass + fuelmass(index - 11500, engine.isp, tmass)]
            else:
                v += [fuelmass(index, engine.isp, engine.mass())]
        if engine.label == 'RL10':
            if index > 10000:
                v += [fuelmass(index - 10000, engine.isp, totalmass(10000, engine.isp, engine.mass()) + engine.weight) + fuelmass(10000, engine.isp, engine.mass())]
            else:
                v += [fuelmass(index, engine.isp, engine.mass())] 
    return np.array(v)


class engine:
    def __init__(self, weight, isp, thrust, power, engines, label):
        self.weight = weight
        self.isp = isp
        self.thrust = thrust
        self.power = power
        self.label = label
        self.engines = engines
        
    def mass(self): #engines + s/c mass + nuc
        return self.weight * self.engines + sc_weight + nucweight(self.power * self.engines, 2)
    
    def massrate(self): #tons / s
        return self.thrust * self.engines / (g0 * self.isp * 1000)
       

engine_list = [
   #engine(weight,  isp,    thrust, power,  engines, label) 
   #engine(0.1,     320,    44000,  0,      1,  'AJ10'),
   #engine(0.25,    470,    66000,  0,      1,  'RL10'),
   #engine(2.2,     950,    80000,  0,      1,  'BNTR'),
    engine(0.05,    4170,   0.25,   7,      1, 'NEXT'),
    engine(0.25,    9600,   0.7,    40,     5,  'HiPEP'),
    engine(0.4,     2900,   2.3,    50,     4,  'AEPS'),
    engine(0.5,     20000,  2.5,    250,    1,  'DS4G'),
    engine(0.68,    5000,   6,      200,    1,  'MPD$_{Ar}$'),
    engine(0.68,    2500,   11,     200,    1,  'MPD$_{Kr}$'),
    engine(2.8,     10141,  20.1,   0,      1,  'DFD') #uses 2000kWe, but weight includes reactor
    ]

sc_weight = 0.4                                                      #ton
drymass_fraction = 0.1                                               #10%
delta_v = np.arange(5000, 17000, 1)

### log plot dv vs mass###
# for engine in engine_list:
#     if(engine.label == 'AJ10' or engine.label == 'RL10'):
#         plot(delta_v, nstage(engine, delta_v) * (1 + drymass_fraction) + engine.mass())
#     else:
#         plot(delta_v, totalmass(delta_v, engine.isp, engine.mass()))  
# plt.grid()
# plt.tight_layout()
# plt.legend()
# plt.gca().set_yscale('log')
# plt.ylabel('Mass (t)')
# plt.xlabel('delta-v (ms$^{-1}$)')
# plt.title('Logarithmic plot of mass vs delta-v')
# plt.show()

### log plot burn time vs deltav ###
for engine in engine_list:
    if(engine.label == 'AJ10' or engine.label == 'RL10'):
        plot(delta_v, nstage(engine, delta_v) / engine.massrate())
    else:
        plot(delta_v, fuelmass(delta_v, engine.isp, engine.mass()) / engine.massrate())
plt.grid()
plt.tight_layout()
plt.legend()
plt.gca().set_yscale('log')
plt.ylabel('Burn time (s)')
plt.xlabel('delta-v (ms$^{-1}$)')
plt.title('Logarithmic plot of burn time vs delta-v')
plt.show() 


### plot dv vs mass per dv###
# for engine in engine_list:
#     if(engine.label == 'AJ10' or engine.label == 'RL10'):
#         y = (nstage(engine, delta_v) + engine.mass()) / delta_v
#     else:
#         y = totalmass(delta_v, engine.isp, engine.mass()) / delta_v        
#     plot(*limitfx(delta_v, y, .01, 0))    
# plt.grid()
# plt.tight_layout()
# plt.legend()
# plt.ylabel('mass / delta-v (t (ms$^{-1}$)$^{-1}$)')
# plt.xlabel('delta-v (ms$^{-1}$)')
# plt.title('Plot of mass / delta-v vs delta-v')
# plt.show()






