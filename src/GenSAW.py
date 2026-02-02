import numpy as np
import matplotlib.pyplot as plt
from SAWFunctions import plotSAW, performMove
import os 
# Slade, Gordon. "Self-avoiding walks." The Mathematical Intelligencer 16.1 (1994): 29-35.
#  Madras, Neal, and Alan D. Sokal. "The pivot algorithm: A highly efficient Monte Carlo method for the self-avoiding walk." Journal of Statistical Physics 50 (1988): 109-186.
#### directions: up: 0, left: 1, down: 2, right: 3

N = 20      # number of particles in SAW
nSample = 5 # number of samples of SAW with N steps

# filename for output
outDir = "data"
outFname = "SAW_N" + str(N) + ".txt"

nEquilibrate = N        # number of accepted equilibration steps
nWaitSample = N      # number of accepted steps before new sample is drawn 

############## initialization ###################

posVec = np.vstack((np.zeros(N),np.linspace(0,N-1, N)))
stepVec = np.ones(N-1) * 0

# output data
allSamples = np.zeros((2*nSample, N))

# discrete probability vectors
nMoves = 7
moveElements = np.linspace(0,nMoves-1,nMoves)
moveProbabilities = np.ones(nMoves) * 1/nMoves
siteElements = np.linspace(1,N-2,N-2) # skip first and last as they correspond to rigid body motions
siteProbabilities = np.ones(N-2) / (N-2)

if os.path.isdir(outDir)==False:
    os.mkdir(outDir)

# counters
iSample = 0
iSinceLastSample = 0

while (iSample < nSample):

    # choose site 
    kSite = np.random.choice(siteElements, 1, p=siteProbabilities)[0]

    # choose move 
    kMove = np.random.choice(moveElements, 1, p = moveProbabilities)

    # identify short end
    if kSite < (N-1)//2:
        chainSide = 0 # left
    else:
        chainSide = 1 # right  

    # execute move
    isUnique, posVec, stepVec = performMove(np.random.choice(np.linspace(0,6,7),1), chainSide, N, kSite, posVec, stepVec)

    # check for sample 
    if isUnique == True and (((iSinceLastSample == nWaitSample-1) and (iSample != 0)) or ((iSinceLastSample == nEquilibrate-1) and (iSample == 0))):

        # save sample 
        originShift = np.vstack([np.repeat(posVec[0,0],N), np.repeat(posVec[1,0],N)])
        allSamples[(2*iSample):(2*(iSample+1))] = posVec - originShift

        # set counters
        iSinceLastSample = 0
        iSample += 1

    elif isUnique == True:
        iSinceLastSample += 1

    np.savetxt(outDir + "/" + outFname, allSamples)
