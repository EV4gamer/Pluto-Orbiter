
import numpy as np
import matplotlib.pyplot as plt

g = 9.81


def massfrac(dv, isp):
    return np.exp(dv/(g * isp))

def massrate(F, isp):
    return F / (isp * g)

#current values for MPD assume a 1MW reactor
#VASMR can go from 1500 - 10000 s isp, but thrust level is not listed for all values
#sc weight does not include tank drymass

sc_weight = 500
th_weight = [100, 250, 2200, 50, 2600, 2600]
isp = [300, 450, 950, 4200, 5000, 1000]
thrust = [44000, 66000, 80000, 0.25, 20, 100]
label = ['HyperGolic', 'LH2', 'NTR-LH2', 'ION', 'MPD', 'MPD m2']
lstyle = ['-', '-', '-', '-', '-', '--']
delta_v = np.arange(5000, 15000, 1)



### log plot dv vs mass###
for i in range(len(isp)):
    plt.plot(delta_v, np.log10((massfrac(delta_v, isp[i]) * (sc_weight + th_weight[i])) / 1000), label = label[i], linestyle=lstyle[i])
    
plt.legend()
plt.ylabel('Mass log$_{10}(t)$')
plt.xlabel('delta-v (ms$^{-1}$)')
plt.show()

### normal plot dv vs mass###
for i in range(len(isp)):
    plt.plot(delta_v, (massfrac(delta_v, isp[i]) * (sc_weight + th_weight[i])) / 1000, label = label[i], linestyle=lstyle[i])
    
plt.legend()
plt.ylabel('Mass (t)')
plt.xlabel('delta-v (ms$^{-1}$)')
plt.show()


### log plot burn time vs deltav ###
for i in range(len(isp)):
    plt.plot(delta_v, np.log10((massfrac(delta_v, isp[i]) * (sc_weight + th_weight[i])) / massrate(thrust[i], isp[i])), label = label[i], linestyle=lstyle[i])

plt.legend()
plt.ylabel('Burn time log$_{10}(s)$')
plt.xlabel('delta-v (ms$^{-1}$)')
plt.show() 

### normal plot burn time vs deltav ###
for i in range(len(isp)):
    plt.plot(delta_v, (massfrac(delta_v, isp[i]) * (sc_weight + th_weight[i])) / massrate(thrust[i], isp[i]), label = label[i], linestyle=lstyle[i])
    
plt.legend()
plt.ylabel('Burn time (s)')
plt.xlabel('delta-v (ms$^{-1}$)')
plt.show() 
