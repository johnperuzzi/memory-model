import time
import numpy as np
import random
neursep = 75 #[um]
numneu = 1000
sepvar = 10 #variability in separation from random placement
c_d = .5 #decay constant for connectivity


def positions(neurons,l=0.5):
	return np.random.uniform([-l,-l,-l],[l,l,l],[neurons,3]),l


def genvec(sep,dim):
	vec = []
	#print('Total number of neurons = %d' % dim**3)
	vals = np.arange(0,dim*sep,sep)
	for i in range(0,dim):
		for j in range(0,dim):
			for k in range(0,dim):
				x = vals[i] #+random.uniform(0,sepvar)
				y = vals[j] #+random.uniform(0,sepvar)
				z = vals[k] #+random.uniform(0,sepvar)
				vec.append([x,y,z])
	return vec

def dist(v1,v2,l):
	d = v1 - v2
	for j in range (3):
		if (d[j] > l/2):
			d[j] -= l
		if (d[j] <= -l/2):
			d[j] += l
	return np.sqrt(np.linalg.norm(d))

# 	dist = 0
# #periodic BC's-- distance from midpoint is a proxy for distance
# 	mid = ((dim-1)*sep+sepvar)/2
# 	for i in range (0,len(v1)):
# 		dist += (abs(v1[i]-mid)-abs(v2[i]-mid))**2
# 	return np.sqrt(dist)

def distmat(pos,l):
	dim = pos.shape[0]
	mat = np.zeros((dim,dim))
	for i in range(dim):
		for j in range(dim):
			mat[i,j] = dist(pos[i,:],pos[j,:],l)
	return mat


	# numneu = len(vec)
	# mat = np.zeros([num, num])
	# for i in range(0,num):
	# 	for j in range(0,num):
	# 		mat[i,j] = dist(vec[i],vec[j],dim,sep)
	# return mat

def connected(dist):
	mm = dist/1000 #distances in mm a la Ebert
	p_connect = np.exp(-mm/c_d)
	if random.uniform(0,1)<=p_connect:
		return 1
	return 0

def connectmat(num, distmat):
	conmat = np.zeros([num, num])
	for i in range(0,num):
		for j in range(0,num):
			if i!=j:
				conmat[i,j] = connected(distmat[i,j])
	return conmat

def wtmat(num,conmat,cMax):
	wts = np.zeros([num, num])
	for i in range (num):
		for j in range(num):
			if conmat[i,j] ==1:
				wts[i,j] = np.random.uniform(high = cMax)
	for i in range (num):
		wts[i,i] = 0
	return wts

def lemmegetwts(num,cMax):
	pos,l = positions(num)
	dists = distmat(pos,l)
	conmat = connectmat(num,dists)
	wts = wtmat(num,conmat,cMax)
	return wts
# tvec = []
# convec = []
# ntrials = 20
# for i in range (0, ntrials):
# 	start_time = time.time()
# 	dim = int(round(np.cbrt(numneu)))
# 	vec = genvec(neursep, dim)
# 	mat = distmat(vec,dim,neursep,numneu)
# 	connections = connectmat(numneu, mat)
# 	sumconnects = 0
# 	ind = random.randint(0,1000)
# 	for i in range (0,numneu):
# 		sumconnects += connections[ind,i]
# 	convec.append(sumconnects)
# 	tvec.append(time.time() - start_time)

# print(("Synapses per neuron = %d +/- %d"), (np.mean(convec), np.std(convec)))
# print(("Run time = %d +/- %d"), (np.mean(tvec), np.std(tvec)))

# print("Total synapses for the %dth neuron: %d" % (ind, sumconnects))
# print("--- %s seconds ---" % (time.time() - start_time))





