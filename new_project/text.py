import math

import matplotlib.pyplot as plt
from math import log10,pi,log2,exp,atan
import numpy as np
fc = 2.4*(10**9)
c = 3*(10**8)


P = 10*log10(2.1)
print(P)
Pdb = 20


List = []
List1 = []
mi = 10
ma = 2000
dd = range(mi,ma)

a = [4.88,9.61,12.08,27.23]
b = [0.43,0.16,0.11,0.08]
n_LoS = [0.1,1,1.6,2.3]
n_NLoS = [21,20,23,34]
env_num = 1
mid_a = a[env_num]
mid_b = b[env_num]
mid_n_LoS = n_LoS[env_num]
mid_n_NLoS = n_NLoS[env_num]
for d in dd:
    PL = 20 * log10(d) + 20 * log10(fc) + 20 * log10(4 * pi / c)
    PLS = 20 * log10(d) + 20 * log10(fc) + 20 * log10(4 * pi / c) + mid_n_NLoS
    PLS += (mid_n_LoS - mid_n_NLoS) / (1 + mid_a * exp(-1 * mid_b * (atan(20 / 20) - mid_a)))

    re = P - PL
    noise = -120
    snr = re - noise
    #snr = 10**(snr/10)
    #ca = log2(1+ snr)
    List.append(snr)
    List1.append(PL)
    re = P - PLS
    snr = re - noise
    snr = 10 ** (snr / 10)
    ca = log2(1 + snr)
    # List1.append(0)

    # List_Pl2.append(20 * log10(d) + 20 * log10(60*fc) + 20 * log10(4 * pi / c))
y = List
y1 = List1
dt = [i for i in dd]
plt.grid()
plt.plot(dt,y,dt,y1)

plt.show()
plt.close()
