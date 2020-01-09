from helper_2 import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


fig, ax1= plt.subplots()
fig, ax2= plt.subplots()

# cExcMaxArray = [100,200,400]
# for cExcMax in cExcMaxArray:
# 	cExcInitArray = [0, int(cExcMax/2), cExcMax]
# 	for cExcInit in cExcInitArray:
exfilename ='/Users/vamsijv/Documents/Vamsi/School/College/Class/Junior Year/Spring/PHYSICS 113/Project/izhikevich_model/example_spiketrain.npy'
filename = '/Users/vamsijv/Documents/Vamsi/School/College/Class/Junior Year/Spring/PHYSICS 113/Project/izhikevich_model/#neur=1000_t=1000_dt=0.5002501250625313_spikedata.npy' 
spiketimes = filename 
# label = 'Max='+ str(int(cExcMax)) +', Init='+ str(int(cExcInit))
kuramotoOP = calcKuramoto(spiketimes,0,1000,1000,np.arange(0,1000),'kOP')
rates = firing_rates(spiketimes,1000,1000)
ax1.plot(rates[0,:-1], rates[1,:-1])
ax2.plot(.001*kuramotoOP[:,0],kuramotoOP[:,1])
		# mswdata = np.load(filename + '/meanWeightTimeSeries_FinalBackup.npy')
		# t = mswdata[:,0]/1000
		# msw_stn = mswdata[:,1]
		# ax2.plot(t,msw_stn, label=label)
		# meansynwt = msw(filename + '/meanWeightTimeSeries_FinalBackup.npy',ax2)
# fontP = FontProperties()
# fontP.set_size('small')
ax1.set_xlabel('Neuron Index')
ax1.set_ylabel('Firing rate (sec^-1)')
ax1.set_title('Neuron Firing rates')
# # Shrink current axis by 20%
# box = ax1.get_position()
# ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
# ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop=fontP)

# ax2.set_xlabel('Time(sec)')
# ax2.set_ylabel('Mean Synaptic Weight')
# ax2.set_title('Mean Synaptic Weight Time Series')
# ax2.legend()
plt.show()
# spiketrain = load_and_shape(filename + '/spikeTimes_FinalBackup.npy')
# plot_msw(filename + 'meanWeightTimeSeries_FinalBackup.npy')	
# plot_events(spiketrain, 50)
# print_gap_stats(spiketrain)