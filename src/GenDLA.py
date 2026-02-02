import numpy as np
import matplotlib.pyplot as plt
import os 
import random

# Algorithm presented by Witten and Sander

#### directions: up: 0, left: 1, down: 2, right: 3

N = 20      # number of particles in DLA
nSample = 5 # number of samples of DLA with N particles

escapeScale = 3 # escapeScale * (largest distance from origin of current particles) sets the distance at which particles can escape
seedOffset = 10 # (largest distance from origin of current particles) + seedOffset sets the seed radius

# filename for output
outDir = "data"
outFname = "DLA_N" + str(N) + ".txt"

############## initialization ###################

# output data
allSamples = np.zeros((2*nSample, N))

if os.path.isdir(outDir)==False:
    os.mkdir(outDir)

# relative position vector array 
latticeVectors = np.array([[0,1], [-1,0], [0,-1], [1,0]])

############## run DLA ###################
for iSample in range(0,nSample):

    print("Sample {:d}".format(iSample))

    # poisition vector
    posVec = np.zeros((2,N))

    # first particle
    posVec[0:2, 0] = [0,0]

    # initialize reference, seed, and escape radii
    Rref = 1 
    Rseed = seedOffset
    Rescape = escapeScale * Rseed

    for iparticle in range(1,N):

        # update reference, seed, and escape radii 
        if iparticle % seedOffset == 0:
            Rref = np.max( np.linalg.norm(posVec, axis=0) )
            Rseed = Rref + seedOffset
            Rescape = np.max( [ Rseed + 2 * seedOffset, escapeScale * Rref])

        # place particle
        theta = 2*np.pi * np.random.rand(1)
        seedLoc = Rseed * (np.cos(theta) * np.array([0,1]) + np.sin(theta) * np.array([1,0]))

        # find closest location on integer grid 
        posVec[0:2, iparticle] = seedLoc.astype(int)

        # random walk
        foundParticleNeighbor = False
        while foundParticleNeighbor == False:

            # update position
            posVec[0:2, iparticle] += latticeVectors[random.randint(0,3)]

            # if particle escaped, initialize again 
            if np.linalg.norm(posVec[0:2, iparticle]) >= Rescape:
                theta = 2*np.pi * np.random.rand(1)
                seedLoc = Rseed * (np.cos(theta) * np.array([0,1]) + np.sin(theta) * np.array([1,0]))
                posVec[0:2, iparticle] = seedLoc.astype(int)

            # check whether there is a neighboring particle already
            dists = np.linalg.norm(posVec - np.reshape(posVec[0:2, iparticle],(2,1)), axis = 0)
            if np.any( np.abs(dists - 1) < 10 * np.finfo(float).eps ):
                foundParticleNeighbor = True
            
    # save sample 
    allSamples[(2*iSample):(2*(iSample+1))] = posVec

np.savetxt(outDir + "/" + outFname, allSamples)
