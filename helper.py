#modified from code written by Ethan Shen (ezshen[at]stanford.edu)


import numpy as np
import matplotlib.pyplot as plt


def plot_msw (filename):
    data = np.load(filename)
    t = data[:,0]/1000
    msw_stn = data[:,1]
    plt.title('Mean Synaptic Weight Time Series')
    plt.xlabel('Time (sec)')
    plt.ylabel('Mean Synaptic Weight')
    plt.plot(t,msw_stn)
    plt.show()


def load_and_shape(filename):
    data = np.load(filename)
    print(data.shape)

    start = data[0][1] # start timepoint
    N = int(max(data[:, 0]) - min(data[:, 0]) + 1) # num neurons
    data = np.array([data[:, 0], data[:, 1] - start]).T # time start at index 0

    lens = []
    for neuron in range(N):
        lens.append(len(data[np.where(data[:, 0] == neuron)][:, 1]))

    num_fires = min(lens) # number of fires for the neuron with the minimium fires (for cutoff)
    #assert max(lens) - num_fires < 10, 'Min number of neuron fires too low!' # check if significantly different from max neuron so don't cut too much off
    firings = np.zeros((N, num_fires)) # (N, T) np array of firings
    for neuron in range(N):
        firings[neuron] = data[np.where(data[:, 0] == neuron)][:, 1][:num_fires]

    return firings

# pass in number of neurons to plot
def plot_events(firings, n,tot):
    plt.title('Neuronal Firings')
    plt.xlabel('Time')
    plt.ylabel('Sample Number')
    n_samples = firings[np.random.choice(range(tot-1), n)] # chooses n neurons to sample for plotting
    plt.eventplot(n_samples)
    plt.show()

##################################################################################################
#             returns array of time in ms (first) and population av Kuramoto order parmameter (second)
#              spikeTimes  ... spike train
#             tmin ... start time of kuramoto time trace in seconds
#             tmax ... end time of kuramoto time trace in seconds
#             NinterPolSteps .. number of equ spaced interpolation steps for time trace
#             arrayOfNeuronIndixes ... speciefies all neurons that are considered for calculation of Kuramoto order parameter
#             outputFilename ... saves data in outputFilename+'.npy'

def calcKuramotoOrderParameter( spikeTimes, tmin, tmax, NinterPolSteps, arrayOfNeuronIndixes, outputFilename ):

    # delete empty entries
    #print tmin, tmax
    spikeTimes=spikeTimes[ spikeTimes[:,1]!= 0 ]
    populationSize=len(arrayOfNeuronIndixes)
    #print populationSize
    #print NinterPolSteps
    phases=np.zeros( (populationSize, NinterPolSteps) )

    arrayOfGridPoints=np.linspace(1000.0*tmin,1000.0*tmax,NinterPolSteps)
    krecPhases=0
    for kNeuron in arrayOfNeuronIndixes:
        # get spike train of corresponding neuron
        spikeTrainTemp=spikeTimes[spikeTimes[:,0]== kNeuron ][:,1]
        #print 'neuron', kNeuron
        #print spikeTrainTemp
        # calc phase function
        if len(spikeTrainTemp) != 0:
                phaseNPiCrossings=np.concatenate( ( np.full( 1, 1000.0*(2*tmin-tmax) ) , spikeTrainTemp, np.full( 1, 1000.0*(2*tmax-tmin) ) ), axis=0 )
                PhaseValues=np.linspace(0,len(phaseNPiCrossings)-1, len(phaseNPiCrossings))
        else:
                phaseNPiCrossings=np.array([-np.inf, np.inf])
                PhaseValues=np.array([0, 1])
        # linear interpolate phaseNPiCrossings
        phaseFunctionKNeuron=interp1d(phaseNPiCrossings,2*np.pi*PhaseValues)
        #print phases.shape, arrayOfGridPoints.shape
        phases[krecPhases,:]=phaseFunctionKNeuron(arrayOfGridPoints)
        krecPhases+=1
        #print '####', kNeuron
        #print phaseNPiCrossings.min(), phaseNPiCrossings.max()
        #print arrayOfGridPoints.min(), arrayOfGridPoints.max()
        #print phases.min(), phases.max()
    # calc Kuramoto order parameter
    TotalArrayOfKuramotoOrderParameterAtGridPoints=1/float(populationSize)*np.absolute(np.sum( np.exp( 1j*phases ), axis=0 ))
    #print 'Kuramoto order parameter'
    #print TotalArrayOfKuramotoOrderParameterAtGridPoints
    KuramotoOutArray=np.array( [arrayOfGridPoints, TotalArrayOfKuramotoOrderParameterAtGridPoints] )
    #print 'phases'
    #print arrayOfGridPoints
    #print TotalArrayOfKuramotoOrderParameterAtGridPoints
    KuramotoOutArray=np.transpose( KuramotoOutArray )
    np.save( outputFilename+'.npy' , KuramotoOutArray )
    return KuramotoOutArray

def print_gap_stats(firings):
    gaps = np.diff(firings)

    # index i is the variance of the spikes for neuron i
    variances = [np.around(np.var(i), decimals=2) for i in gaps]

    # index i is the average spike gap for neuron i
    avggaps = [np.around(np.mean(i), decimals=2) for i in gaps]

    # neuron with the min and max average gap
    print("Min avg gap:\n", min(avggaps))
    print("Max avg gap:\n", max(avggaps))

    # normalized gaps
    gap_norm = []
    for i in range(len(gaps)):
        # take gaps[i] -= mean of gaps[i]
        gap_norm.append([gaps[i][j]-avggaps[i] for j in range(1,len(gaps[i]))])

    # index i of avg_err_p is abs value of gap_norm[i]/gap_mean[i] for neuron i
    # intuitively, entry i is average percent deviation from pattern of neuron i
    avg_err_p = [np.mean(np.abs(gap_norm[i]))/avggaps[i] for i in range(len(gap_norm))]

    # this is average across neurons of average percent deviation of each neuron
    print("avg % err\n", np.mean(avg_err_p))

    # index i is average number of time steps off neuron i is from it's pattern
    avg_err = [np.mean(np.abs(gap_norm[i])) for i in range(len(gap_norm))]
    print("avg err\n", np.mean(avg_err))

    # avg across neurons of standard deviation of the times for that neuron
    print("avg std of times:\n", np.mean(np.sqrt(variances)))
    # avg across neurons of average deviation time from pattern
    print("std of avg diffs:\n", np.sqrt(np.var(avg_err)))

