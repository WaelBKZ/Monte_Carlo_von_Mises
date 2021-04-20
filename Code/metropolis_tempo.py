from math import *
import numpy as np
from tools import *

#règle simple = acceptation rate entre 25 et 40 ?

def pi_func(x,kappa=3,mu=0):
    return exp(kappa*cos(x-mu)) * int((-pi<=x<=pi))

def metropolis_rw(x,sig=0.1):
    y = x + sig*np.random.randn()
    r = pi_func(y)/pi_func(x)
    u = np.random.rand()
    if u<r:
        return (y,1)
    else :
        return (x,0)

N=100_000
x=np.empty(N)
l=0
kappa=3
sig=3*1/sqrt(2*kappa)
sig=2
for i in range(1,N):
    result=metropolis_rw(x[i-1],sig)
    x[i]=result[0] #sigma trop grand : on est constant par morceaux
    l+=result[1]
print(l/N)

import matplotlib.pyplot as plt
# # plt.plot(x)
# # plt.show()
#
import seaborn as sb
sb.distplot(x)
plt.show()

mu = 0
kappa = 3
n = 100_000
from tools import *
from scipy.stats import vonmises
data = x

fig, ax = plt.subplots(1, 1)
ax.hist(data, bins=500, density=True)

x = np.linspace(vonmises.ppf(0.01, kappa, loc=mu), vonmises.ppf(0.99, kappa, loc=mu), 100)
ax.plot(x, vonmises.pdf(x, kappa, loc=mu), 'r-', lw=1, label='theoretical')
ax.set_title('Von Mises simulation')
plt.legend()
plt.show()