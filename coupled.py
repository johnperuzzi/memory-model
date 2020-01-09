import numpy as np
from matplotlib import pyplot as plt
import random
import neuron_module as nm

## Contains methods to simulate an indivudal connection between neurons as per Izhikevich, IEEE Transactions on Neural Networks (2003) 14:1569- 1572

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
tf = 500 #msec
v0 = c #mV
u0 = 0
n =2000
v_e = -70 #reversal potential of presynaptic neuron (mV)
tau = 2 #characteristic time of PSP decay (msec)
pf = 10 #exponential prefactor for PSP
t = np.linspace(t0,tf,n)

#step size
dt = (tf-t0)/(n-1)

#####################################################################
# Presynaptic Neuron #
#####################################################################

#time and value vectors
v1 = np.zeros([n]) 
u1 = np.zeros([n])
v1[0] = v0
u1[0] = u0

#####################################################################
# Postsynaptic Neuron #
#####################################################################

#time and value vectors
v2 = np.zeros([n]) 
u2 = np.zeros([n])
g2 = np.zeros([n])

phase = random.randint(1,100) # arbitrary shift between neurons

v2[0] = v0
u2[0] = u0


#####################################################################
# Simulation #
#####################################################################

for i in range(1,n):
	I1 = I_0
	#coupling between the two neurons
	I2 = I_0/2 + g2[i-1]
	
	v1[i] = nm.v_step(dt, v1[i-1], u1[i-1], I1)
	u1[i] = nm.u_step(dt, v1[i-1], u1[i-1], I1)
	
	g2[i] = dt * ((-1*g2[i-1])/tau) + g2[i-1]
	v2[i] = nm.v_step(dt, v2[i-1], u2[i-1], I2)
	u2[i] = nm.u_step(dt, v2[i-1], u2[i-1], I2)

	#reset after spiking event and feed into post
	if (v1[i-1] >= spikethresh):
		v1[i] = c
		u1[i] = u1[i] + d
		g2[i] = pf

	if (v2[i-1] >= spikethresh):
		v2[i] = c
		u2[i] = u2[i] + d
		
#clip peaks to make graph look pretty
spikes = np.zeros((len(v2),))
for i in range (n):
	if (v1[i] >=spikethresh):
		v1[i] = spikethresh
	if (v2[i] >=spikethresh):
		v2[i] = spikethresh
		spikes[i] = 1.0


# plt.plot(np.blackman(len(spikes)))
# plt.title('Blackman Window')

fig1,ax = plt.subplots(2,2,figsize=(20,9))
ax1,ax2,ax3,ax4 = ax.flatten()

ax1.set_title("Postsynaptic Membrane Voltage") 
ax2.set_title("Presynaptic Membrane Voltage") 
ax1.plot(t, v2, label = 'Postsynaptic', c='g')
ax2.plot(t, v1, label = 'Presynaptic',c='m')
ax2.set_xlabel("Time (ms)")
ax1.set_xlabel("Time (ms)")
ax1.set_ylabel("Membrane potential (mV)")
ax2.set_ylabel("Membrane potential (mV)")
fig1.suptitle("Two Coupled Neurons") 
fig1.legend()


ax3.plot(v2, u2, label = 'Postsynaptic', c='g')
ax3.plot(v1, u1, label = 'Presynaptic', c='m')
ax3.set_xlabel("Membrane potential (mV)")
ax3.set_ylabel("Recovery Variable (pA)")
ax3.set_title("Phase Plot") 


# FFT of spike train
# fig3,ax4 = plt.subplots()

spikes = spikes * np.blackman(len(spikes)) #filter with blackman window
sfft=np.fft.fft(spikes, len(spikes)) # execute fft
sfreqs=np.fft.fftshift(np.fft.fftfreq(len(sfft)))
# filtered = (np.abs(sfft))*(np.abs(sfreqs)<=0.05).astype(float) #rect filter
filtered = np.abs(sfft) 
mid = int(len(sfft)/2)
ax4.plot(1000*sfreqs[mid:],filtered[mid:], c='g')
ax4.set_xlabel("Frequency (Hz)")
ax4.set_ylabel("Intensity")
ax4.set_title("Postsynaptic Frequency Spectrum") 
# ax2.plot(autocorr(spikes))

plt.show()
