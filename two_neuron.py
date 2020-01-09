import numpy as np
from matplotlib import pyplot as plt
import random
import neuron_module as nm


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
g0 = 0
n =2000
v_e = -70 #reversal potential of presynaptic neuron (mV)
tau = 5 #characteristic time of PSP decay (msec)



#step size
dt = (tf-t0)/(n-1)

#coupling matrix

w = np.mat('0,1; .5,0')
neurons = w.ndim;

#value mat: [number of neurons]x3 dim matrix, where col0 = v, col1 = u, 
#col2 = g for the ith neuron
valmat = np.zeros([neurons, 3])
phase = np.random.randint(1, 20, size =neurons)

#spike catalog
spikemat = np.zeros([neurons,n])

#####################################################################
# Plot Two Firing neurons #
#####################################################################

plt.figure()
plt.axis([0, n, -5, spikethresh+5])
plt.xlabel("Time (ms)")
plt.ylabel("Membrane potential (mV)")
plt.title("Two Coupled Neurons") 
plt.ion()

for t in range (1, n):
	for i in range (1, neurons):
		if t == phase[i]:
			valmat[i,0]=v0 
			valmat[i,1]=u0
			valmat[i,2]=g0
		valvec = valmat[i-1,:]
		nm.neurstep(dt, valvec, I_0, spikemat[:,t], w, i, neurons, tau, v_e)
		valmat[i-1,:] = valvec
	tvec = np.full([neurons,1], t)
	try:
		plt.plot(tvec, valmat[:,0])
		plt.draw()
		plt.pause(0.001)
	except KeyboardInterrupt:
		plt.close('all')
		sys.exit('0')
		

# np.save('2neurspike', spikemat)
# norm = (np.linalg.norm(spikemat))**2
# print(norm)


