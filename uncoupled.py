import numpy as np
from matplotlib import pyplot as plt
import random
import neuron_module as nm

## Contains methods to simulate an indivudal neuron as per Izhikevich, IEEE Transactions on Neural Networks (2003) 14:1569- 1572

# inject current here
I_0 = 10 #mV

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
t = np.linspace(t0,tf,n)

#step size
dt = (tf-t0)/(n-1)

#####################################################################
# Presynaptic Neuron #
#####################################################################

#time and value vectors
v1 = np.zeros([n]) 
u1 = np.zeros([n])
g1 = np.zeros([n])
v1[0] = v0
u1[0] = u0


nm.single_neuron(dt, n, 1, v1, u1, nm.v_step, nm.u_step, I_0, 0)

#####################################################################
# Postsynaptic Neuron #
#####################################################################

#time and value vectors
v2 = np.zeros([n]) 
u2 = np.zeros([n])
g2 = np.zeros([n])

phase = random.randint(1,100) # arbitrary shift between neurons

v2[phase] = v0
u2[phase] = u0

nm.single_neuron(dt, n, phase, v2, u2, nm.v_step, nm.u_step, I_0, 0)


plt.plot(t, v1)
plt.plot (t, v2)
plt.xlabel("Time (ms)")
plt.ylabel("Membrane potential (mV)")
plt.title("Two Uncoupled Neurons") 
plt.show()
plt.close()