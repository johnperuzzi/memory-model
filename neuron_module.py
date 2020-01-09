import numpy as np
from matplotlib import pyplot as plt

# Contains methods to simulate an indivudal neuron as per Izhikevich, IEEE Transactions on Neural Networks (2003) 14:1569- 1572



#neuronal dynamics variables
a = 0.02
b = 0.2
c = -61
d = 2
spikethresh = 30


#################################
# Forward Euler step in voltage variable # 
#################################

def v_step(dt, v_last, u_last, I):
	v =  dt * ((0.04*v_last*v_last) + 5*v_last + 140 - u_last + I) + v_last
	return v

#################################
# Forward Euler step in recovery variable # 
#################################
def u_step(dt, v_last, u_last, I):
	u = dt * (a*(b*v_last - u_last)) + u_last
	return u

#################################
# Step in synaptic conductance #
#################################

def sc_step(dt, g_last, w, index, spikevec, neurons, tau):
	coupling = w[index, :]
	coupling = coupling.reshape(1,neurons)

	PSP= np.dot(coupling, spikevec) # set one as row, other as column
	g = g_last + (dt/tau)*(-g_last + PSP/neurons)
	# print(g)

	return g

#use only for mutliple neurons
def neurstep(dt, valvec, I_0, oldspikevec, newspikevec, w, index, neurons, tau, v_e, chatter=False):
	#neuronal dynamics variables
	a = 0.02
	b = 0.2
	c = -61
	d = 2
	spikethresh = 30

	#Euler method to solve coupled DEs
	g_i = sc_step(dt, valvec[2], w, index, oldspikevec, neurons, tau)
	valvec[2] = g_i
	I_syn = g_i*(v_e - valvec[0])

	#reset after spiking event
	if (valvec[0] >= spikethresh):
		if chatter:
			c = - 50 # [mV]
			d  = 2
		valvec[0] = c
		valvec[1] = valvec[1] + d
		return False
	else:
		I = I_0 + I_syn
		# print(I)
		valvec[0] = v_step(dt, valvec[0], valvec[1], I)
		valvec[1] = u_step(dt, valvec[0], valvec[1], I)
		if (valvec[0] >=spikethresh):
			valvec[0] = spikethresh
			newspikevec[index] = 1
			return True
	return False


def single_neuron():
	
	#initial conditions
	t0 = 0
	tf = 1000 #msec
	v0 = c #mV
	u0 = 0
	n =2000 

	#step size
	dt = (tf-t0)/(n-1)

	#time and value vectors
	t = np.linspace(t0,tf,n)
	valmat = np.zeros([2, n]) 
	valmat[0,0] = v0
	valmat[1,0] = u0

	for i in range (1,n):
		# inject current here
		I = 10 #mV
		valvec = valmat[:, i]
		neurstep(dt, valvec, I, 0, 0, 0, 0, 1, 0, 0)
	plot_activity(t, valmat[0,:])

def plot_activity(t, v):
	plt.plot(t, v)
	plt.xlabel("Time (ms)")
	plt.ylabel("Membrane potential (mV)")
	plt.title("Single Neuron Model-- Constant Current") 
	plt.show()
	plt.close()

