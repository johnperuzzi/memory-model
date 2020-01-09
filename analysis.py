from helper import *

#example
exfilename ='/Users/vamsijv/Documents/Vamsi/School/College/Class/Junior Year/Spring/PHYSICS 113/Project/izhikevich_model/example_spiketrain.npy'
# filename = '/Users/vamsijv/Documents/Vamsi/School/College/Class/Junior Year/Spring/PHYSICS 113/Project/izhikevich_model/#neur=1000_t=1000_dt=0.5002501250625313_spikedata.npy'
# filename ='/Users/vamsijv/Documents/Vamsi/School/College/Class/Junior Year/Spring/PHYSICS 113/Project/izhikevich_model/#neur=1000_t=500_dt=0.5005005005005005_spikedata.npy'
# exdata = np.load(exfilename)
# data = np.load(filename)
# np.savetxt('exdata.csv', exdata)
# np.savetxt('data.csv', data)

for i in range(4):
	filename = '/Users/vamsijv/Documents/Vamsi/School/College/Class/Junior Year/Spring/PHYSICS 113/Project/izhikevich_model/'+str(100+10*i)+'_spikedata.npy'
	spiketimes = filename 
	spiketrain = load_and_shape(spiketimes)
	plot_events(spiketrain, 50,1000)
	print_gap_stats(spiketrain)