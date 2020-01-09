import numpy as np
from matplotlib import pyplot as plt
import random

# Contains methods to simulate a single connection between two indivudal neurons as per Izhikevich, IEEE Transactions on Neural Networks (2003) 14:1569- 1572

# inject current here
I_0 = 5 #mV

#neuronal dynamics variables
a = 0.02
b = 0.2
c = -61
d = 2
spikethresh = 30

#initial conditions
t0 = 0 
tf = 1000 #msec
v0 = c #mV
u0 = 0
n =10000
v_e = -70 #reversal potential of presynaptic neuron (mV)
tau = 2 #characteristic time of PSP decay (msec)
pf = 10 #exponential prefactor for PSP

#step size
dt = (tf-t0)/(n-1)

#time and value vectors
t = np.linspace(t0,tf,n)
v = np.zeros([n]) 
u = np.zeros([n])
g = np.zeros([n]) 
v[0] = v0
u[0] = u0
g[0] = pf

#Euler method to solve coupled DEs
for i in range(1,n):
	I = I_0 + g[i-1]
	if (random.randint(1,10) == 1):
		g[i] = pf
	else:
		g[i] = dt * ((-1*g[i-1])/tau) + g[i-1]
	#reset after spiking event
	if (v[i-1] >= spikethresh):
		v[i] = c
		u[i] = u[i] + d
	else: 
		v[i] = dt * ((0.04*v[i-1]*v[i-1]) + 5*v[i-1] + 140 - u[i-1] + I) + v[i-1]
		u[i] = dt * (a*(b*v[i-1] - u[i-1])) + u[i-1]

#reset after spiking event
	if (v[i-1] >= spikethresh):
		v[i] = c
		u[i] = u[i] + d

#clip peaks to make graph look pretty
for i in range (n):
	if (v[i] >=spikethresh):
		v[i] = spikethresh


plt.plot(t, v)
plt.plot(t, g)
plt.xlabel("Time (ms)")
plt.ylabel("Membrane potential (mV)")
plt.title("Single Neuronal Connection") 
plt.show()
plt.close()