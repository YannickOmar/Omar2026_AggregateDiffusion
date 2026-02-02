import numpy as np
from KR_RPY_utils import scaleParticleLocs, computeRH, KirkwoodRisemanSolve, compHullSampling
import os 
import itertools
import scipy as sc
import sys


def readKeyValueFile(filePath):
    """
    Reads a file where odd lines contain keys and even lines contain values,
    and returns a dictionary mapping each key to its corresponding value.
    """
    dataDict = {}
    with open(filePath, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]  # strip whitespace & ignore blank lines

    # Iterate through lines two at a time
    for i in range(0, len(lines), 2):
        key = lines[i]
        value = lines[i + 1] if i + 1 < len(lines) else None
        dataDict[key] = value

    return dataDict

    
#################### get input ####################
if len(sys.argv) < 2:
    print("Usage: python script.py <inputFile>")
    sys.exit(1)

# get input file 
inputFile = sys.argv[1]  # first argument after the script name
    
# read input 
inputData = readKeyValueFile(inputFile)

#################### init input ####################
# input directory 
inputDir = inputData['INPUT_DIR']

# input file extension
inputExtension = ".txt"

# output directory
outputDir = inputData['OUTPUT_DIR']

# output file name ending
outputFnameEnding = "_diffCoeff.txt"

# membrane viscosity [pN mus/nm]
zeta = float(inputData['ZETA'])

# bulk viscosity [pN mus/nm^2]
eta = float(inputData['ETA'])

# distance from wall [nm]
h = float(inputData['H'])

# particle radius [nm]
particleR = float(inputData['PARTICLE_RADIUS'])

# bond length [nm]
bondLength = float(inputData['BOND_LENGTH'])

# size of original lattice site
latticeDist = 1

# compute convex(SD)/concave(ES) hull
compCHull = True

# method to use for approximation 0: convex hull, 1: concase hull, 2: buffer method
approxMethod = float(inputData['APPROXIMATION_METHOD'])

# parameter used in approximating geometry
approxFactor = float(inputData['APPROXIMATION_FACTOR'])

# number of repetitions for convex hull
cHullNRep = 10

print("---- parsed input data ----")

#################### initializations ####################
# create output directory 
if os.path.isdir(outputDir)==False:
    os.mkdir(outputDir)

# find all input files
inputFiles = []
for file in os.listdir(inputDir):
    if file.endswith(inputExtension):
        inputFiles.append(os.path.join(inputDir, file))


if h == float('inf'):
    ell = zeta/(2*eta)      # Saffmann-Delbrueck lengthscale
else:
    ell = np.sqrt(h * zeta/eta)     # Evans-Sackmann lengthscale
    
# single particle drag coefficient 
if h == float('inf'):
    dragCoeff = 4 * np.pi * zeta /(np.log(2*ell/particleR) - np.euler_gamma) # orginal SD
else:
    epsR = particleR/ell
    dragCoeff = 4 * np.pi * zeta * (1/4 * epsR**2 + epsR * sc.special.kv(1,epsR)/sc.special.kv(0,epsR) )

# counters
fileCount = 0

#################### computations ####################

# loop over all input files
for cFilename in inputFiles:

    print("Processing file {:s}".format(os.path.basename(cFilename)))
    fileCount += 1

    # number of realizations
    numReal = int( sum(1 for _ in open(cFilename))/2 )

    # output data 
    totalDragCoeffs = np.zeros((numReal,10))

    # realization counter
    iReal = 0

    # open file
    with open(cFilename, "r") as file:

        firstLine = True

        # loop over individual realizations
        for xline,yline in itertools.zip_longest(*[file]*2):
            
            # extract particle locations
            xlocs = np.array(xline.strip().split(), dtype=float) 
            ylocs = np.array(yline.strip().split(), dtype=float) 
            particleLocs = np.vstack([xlocs[~np.isnan(xlocs)],ylocs[~np.isnan(ylocs)]])

            # Note: right-hand side vector is the same for all realizations because number of particles could change
            # number of particle
            Np = particleLocs.shape[1]

            # set flag
            firstLine = False

            # scale position vectors to get correct particle spacing
            particleLocs = scaleParticleLocs(particleR, bondLength, particleLocs, latticeDist)

            # center of mass 
            CoM = np.sum(particleLocs,axis=1)/Np

            # compute particle locations relative to center of mass
            particleLocs = particleLocs  - CoM.reshape(-1, 1)

            # radius of gyration
            Rg = np.linalg.norm( ( particleLocs ) )/np.sqrt(Np) 

            # compute drag coefficient 
            Xixx, Xixy, Xiyy = KirkwoodRisemanSolve(particleLocs.transpose(), eta, zeta, particleR, h, dragCoeff)

            # compute hydrodynamic radii
            RHs, RHl = computeRH(particleLocs, Np, ell, particleR, h, dragCoeff, zeta)

            XixxCHull, XixyCHull, XiyyCHull, geomMRijErrorChull = 0, 0, 0, 0
            if compCHull == True:

                # compute geometric mean of interparticle distance 
                Rij = np.zeros( int(Np * (Np-1)/2) )
                cnt = 0
                for ir in range(Np):
                    for jr in range(0, ir):
                        Rij[cnt] = np.linalg.norm( particleLocs[:,ir] - particleLocs[:,jr] )
                        cnt += 1
                geomMRij = sc.stats.gmean(Rij)


                for iChullRep in range(cHullNRep):
                      
                    XixxApprox, XixyApprox, XiyyApprox, geomMRijSample = \
                    compHullSampling(particleLocs, particleR, bondLength, eta, zeta, h, dragCoeff, ell, approxMethod, approxFactor)  

                    XixxCHull += XixxApprox
                    XixyCHull += XixyApprox
                    XiyyCHull += XiyyApprox
                    geomMRijErrorChull += np.abs( np.log(geomMRijSample) - np.log(geomMRij) )/np.log(geomMRij)
            
                XixxCHull /= cHullNRep
                XixyCHull /= cHullNRep
                XiyyCHull /= cHullNRep
                geomMRijErrorChull /= cHullNRep

            # total drag coefficient
            totalDragCoeffs[iReal,:] = np.array([Rg, RHs, RHl, Np, Xixx, Xixy, Xiyy, XixxCHull, XixyCHull, XiyyCHull])

            # update counter
            iReal += 1

    # write results to file
    basename = os.path.basename(cFilename)
    np.savetxt(os.path.join(outputDir, basename.replace(inputExtension, outputFnameEnding)), totalDragCoeffs)



