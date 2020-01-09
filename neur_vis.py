# Neuron Placement Visualization

from neurmat import *
# import functions_copy as fn
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


pos,l = positions(1000)


#Plot histogram of neuron distances
#changed functions to output distances instead of synReversals
distances = distmat(pos,l)

# #upper diagonal indices
# iu = np.triu_indices(1000)
# plot and load data
fig,(ax1,ax2) = plt.subplots(2)

y = pos[:,0]
x = np.arange(len(y))
counts, bins, bars = ax1.hist(y.T,bins=50)
# plt.show()

ax1.set_xlabel('Distance (mm)')
ax1.set_ylabel('Count')
ax1.set_title('Distribution of Interneuron Distances')

# # Plot neuron distributions 
# fig = plt.figure()
ax2 = Axes3D(fig)

ax2.scatter(pos[:, 0], pos[:, 1], pos[:, 2])

ax2.set_title('Sample Spatial Neuron Distribution')
ax2.set_xlabel('X (mm)')
ax2.set_ylabel('Y (mm)')
ax2.set_zlabel('Z (mm)')
plt.show()