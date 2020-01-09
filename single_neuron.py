import numpy as np
import time 
from matplotlib import pyplot as plt
import scipy.fftpack



# Contains methods to simulate an indivudal neuron as per Izhikevich, IEEE Transactions on Neural Networks (2003) 14:1569- 1572

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[result.size/2:]




# inject current here
I = 10 #mV

#neuronal dynamics variables
a = 0.02
b = 0.2
# regular
c = -65
d = 8
# # chattering
# c = -50
# d = 2
spikethresh = 30

#initial conditions
t0 = 0
tf = 1000 #msec
v0 = c #mV
u0 = 0
n =5000 

#step size
dt = (tf-t0)/(n-1)

#time and value vectors
t = np.linspace(t0,tf,n)
v = np.zeros([n]) 
u = np.zeros([n]) 
v[0] = v0
u[0] = u0

start_euler = time.time()

#Euler method to solve coupled DEs
for i in range(1,n):
	v[i] = dt * ((0.04*v[i-1]*v[i-1]) + 5*v[i-1] + 140 - u[i-1] + I) + v[i-1]
	u[i] = dt * (a*(b*v[i-1] - u[i-1])) + u[i-1]
	#reset after spiking event
	if (v[i-1] >= spikethresh):
		v[i] = c
		u[i] = u[i] + d

veuler = v
ueuler = u


v = np.zeros([n]) 
u = np.zeros([n]) 
v[0] = v0
u[0] = u0

euler_runtime = time.time() - start_euler
print('Euler Runtime = ' + str(euler_runtime))
start_rk = time.time()

# 2nd order Runge-Kutta to solve same DEs as above
for i in range (1,n):
	kv1 = dt * ((0.04*v[i-1]*v[i-1]) + 5*v[i-1] + 140 - u[i-1] + I)
	ku1 = dt * (a*(b*v[i-1] - u[i-1]))

	kv2 = dt * ((0.04*(v[i-1]+0.5*kv1)*(v[i-1]+0.5*kv1)) + 5*(v[i-1]+0.5*kv1) + 140 - (u[i-1]+0.5*ku1) + I)
	ku2 = dt * (a*(b*(v[i-1]+0.5*kv1) - (u[i-1]+0.5*ku1)))

	u[i] = ku2 + u[i-1]
	v[i] = kv2 + v[i-1]
#reset after spiking event
	if (v[i-1] >= spikethresh):
		v[i] = c
		u[i] = u[i] + d

vrk = v
urk = u

#clip peaks to make graph look pretty
spikes = np.zeros((len(v),))
for i in range (n):
	if (v[i] >=spikethresh):
		v[i] = spikethresh
		spikes[i] = 1.0

rk_runtime = time.time()-start_rk
print('RK Runtime = ' + str(rk_runtime))


fig, (ax1,ax2) = plt.subplots(2,sharex=True)
verr = vrk[1:len(vrk)-1] - veuler[1:len(vrk)-1]
uerr = urk[1:len(vrk)-1] - ueuler[1:len(vrk)-1]
ax1.plot(t[1:len(vrk)-1],uerr,label = 'u error',c='g')
ax2.plot(t[1:len(vrk)-1],verr,label = 'v error',c='r')
ax2.set_xlabel('Time(ms)')
ax2.set_ylabel('Error')
ax2.set_xlabel('Time(ms)')
fig.legend()
plt.show()

# fig, ax = plt.subplots(2,2,figsize=(20,9))
# ax1, ax2, ax3, ax4 = ax.flatten()

# fig.suptitle("Single Chattering Neuron") 

# #plot v 
# ax1.plot(t, v)
# ax1.set_xlabel("Time (ms)")
# ax1.set_ylabel("u(mV)")
# ax1.set_title("Membrane Voltage Time Series")


# # FFT of spike train
# # spikes = np.append(spikes,np.zeros(10*len(spikes)))
# spikes = spikes * np.blackman(len(spikes)) #filter with blackman window
# sfft=np.fft.fft(spikes, len(spikes)) # execute fft
# sfreqs=np.fft.fftshift(np.fft.fftfreq(len(sfft)))
# # filtered = (np.abs(sfft))*(np.abs(sfreqs)<=0.05).astype(float) #rect filter
# filtered = np.abs(sfft) 
# mid = int(len(sfft)/2)
# ax2.plot(1000*sfreqs[mid:],filtered[mid:])
# # ax2.plot(autocorr(spikes))

# ax2.set_xlabel("Frequency (Hz)")
# ax2.set_ylabel("Intensity")
# ax2.set_title("Spike Train Frequency Spectrum")

# #plot u
# ax3.plot(t, u)
# ax3.set_xlabel("Time (ms)")
# ax3.set_ylabel("u (pA)")
# ax3.set_title("Recovery Variable Time Series")

# #plot phase plane
# ax4.plot(v,u)
# ax4.set_xlabel("v (mV)")
# ax4.set_ylabel("u (pA)")
# ax4.set_title("Phase Plot")

# plt.show()
