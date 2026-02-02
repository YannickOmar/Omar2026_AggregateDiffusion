import numpy as np
import scipy as sc
import shapely

# compute Oseen tensor analytically
def computeRPYTensor(particles, eta, zeta, particleSize, h):

    # number of interacting particles
    Np = len(particles)

    if h == float('inf'):
        ell = zeta/(2*eta)
    else:
        ell = np.sqrt(h * zeta/eta)

    # finite size factor
    alpha = 1/2 * (particleSize/ell)**2
    
    # array of Oseen tensor components
    RPYTensors = np.zeros((Np, Np, 3))

    for ip in range(0,Np):

        # particle location
        Ri = particles[ip]

        for jp in range(ip+1,Np):
            
            # location of particle j
            Rj = particles[jp]

            # distance 
            RijVec = ( Ri - Rj )/ell
            Rij = np.linalg.norm( RijVec )

            # compute T_ij
            if h == float('inf'):
                H0 = sc.special.struve(0,Rij)
                H1 = sc.special.struve(1,Rij)
                Hm1 = sc.special.struve(-1,Rij)
                Y0 = sc.special.yv(0,Rij)
                Y2 = sc.special.yv(2,Rij)
                
                RPYTensors[ip,jp,0] = 1/(4 * zeta) * ( H0 * (1-alpha) - alpha/Rij * Hm1 - 1/Rij * H1 - (1-alpha)/2 * (Y0 - Y2) + 2/(np.pi * Rij) * (1/Rij + alpha)  - 
                                                         ( H0 * (1-alpha) - 2 * alpha/Rij * Hm1 - 2/Rij * H1 + (1-alpha) * Y2 + 2/(np.pi * Rij) * (2/Rij + alpha)) * RijVec[0]**2/(Rij**2)  )
                
                RPYTensors[ip,jp,1] = -1/(4 * zeta) * ( H0 * (1-alpha) - 2 * alpha/Rij * Hm1 - 2/Rij * H1 + (1-alpha) * Y2 + 2/(np.pi * Rij) * (2/Rij + alpha)) * RijVec[0] * RijVec[1]/(Rij**2) 
                
                RPYTensors[ip,jp,2] = 1/(4 * zeta) * ( H0 * (1-alpha) - alpha/Rij * Hm1 - 1/Rij * H1 - (1-alpha)/2 * (Y0 - Y2) + 2/(np.pi * Rij) * (1/Rij + alpha)  - 
                                                         ( H0 * (1-alpha) - 2 * alpha/Rij * Hm1 - 2/Rij * H1 + (1-alpha) * Y2 + 2/(np.pi * Rij) * (2/Rij + alpha)) * RijVec[1]**2/(Rij**2)  )
                
            else:
                K0 = sc.special.kv(0,Rij)
                K1 = sc.special.kv(1,Rij)


                RPYTensors[ip,jp,0] = 1/(2 * np.pi * zeta) * ( (K0 + 1/Rij * K1) * (1 + alpha) - 1/Rij**2 + ( 2/(Rij**2) - (K0 + 2/Rij * K1) * (1 + alpha) ) * RijVec[0]**2/(Rij**2) )

                RPYTensors[ip,jp,1] = 1/(2 * np.pi * zeta) * ( ( 2/(Rij**2) - (K0 + 2/Rij * K1) * (1 + alpha) ) * RijVec[0] * RijVec[1]/(Rij**2)  )

                RPYTensors[ip,jp,2] = 1/(2 * np.pi * zeta) * ( (K0 + 1/Rij * K1) * (1 + alpha) - 1/Rij**2 + ( 2/(Rij**2) - (K0 + 2/Rij * K1) * (1 + alpha) ) * RijVec[1]**2/(Rij**2)  )
            
            # use symmetry to get other interaction too (should not store this explicitly though)
            RPYTensors[jp,ip,0] = RPYTensors[ip,jp,0]
            RPYTensors[jp,ip,1] = RPYTensors[ip,jp,1]
            RPYTensors[jp,ip,2] = RPYTensors[ip,jp,2]

    return RPYTensors

# assemble Oseen tensor into global interaction matrix
def CreateMat(OseenT, dragCoeff, Np):
    Mat = np.zeros((2*Np, 2*Np))

    for l in range(0, Np):

        Mat[2*l,2*l] = 1
        Mat[2*l+1,2*l+1] = 1
        
        for i in range(0,Np):
            if (i == l):
                continue

            Mat[2*l, 2*i] = dragCoeff * OseenT[i,l,0] 
            Mat[2*l, 2*i+1] = dragCoeff * OseenT[i,l,1] 
            Mat[2*l+1, 2*i] = dragCoeff * OseenT[i,l,1] 
            Mat[2*l+1, 2*i+1] = dragCoeff * OseenT[i,l,2] 
    return Mat


# scale particle locations to appropriate bond and particle size
# new particle distance = 2 * particleR + bondLength
# oldParticlesLocs.shape(): [dim, # particles]
# assumption: all adjacent particles have the same distance between them
def scaleParticleLocs(particleR, bondLength, oldParticleLocs, baseLength):
    
    return oldParticleLocs * (2 * particleR + bondLength)/baseLength

def computeRH(particleLocs, Np, ell, particleR, h, dragCoeff, zeta):
    ## hydrodynamic radius ##
    expDoubleSum = 0
    for ip in range(0,Np):
        for jp in range(0,ip):
            Rij = np.linalg.norm( particleLocs[:,ip] - particleLocs[:,jp] )/ell
            if h == float('inf'):
                expDoubleSum += 2 * ( (sc.special.struve(0,Rij) - sc.special.yv(0,Rij) ) ) 
            else:
                expDoubleSum += 2*sc.special.kv(0, Rij)

    if h == float('inf'):
        RHs = 2*ell * (particleR/(2*ell))**(1/Np) * np.exp(-np.euler_gamma * (Np-1)/Np - np.pi/(2 * Np**2) * expDoubleSum)
        RHl = ell / ( 8*zeta/(Np * dragCoeff) + 1/Np**2 * expDoubleSum )
        
    else:
        epsR = particleR/ell
        muepsR = 1/2 * 1/(1/4 * epsR**2 + epsR * sc.special.kv(1,epsR)/sc.special.kv(0,epsR) )

        RHs = 2 * ell * (particleR/(2*ell))**(1/Np) * np.exp( - 1/Np**2 * (1 - 1/2 * (particleR/ell)**2) * expDoubleSum - np.euler_gamma * (Np-1)/Np)
        RHl = 2 * ell * Np * 1/np.sqrt( 2 * Np * muepsR + (1 - 1/2 * (particleR/ell)**2) * expDoubleSum )

    return RHs,RHl

def createRangeWithMidpoint(minPoint, maxPoint, midPoint, minDist):

    assert( minPoint < midPoint )
    assert( midPoint < maxPoint )

    Nleft = int(np.ceil((midPoint - minPoint)/minDist))
    Nright = int(np.ceil((maxPoint - midPoint)/minDist))

    leftVals = np.linspace( midPoint - Nleft*minDist, midPoint, Nleft+1 ) 
    rightVals = np.linspace( midPoint, midPoint + Nright * minDist, Nright+1 )

    return np.unique(np.concatenate((leftVals, rightVals)))
    

def placeParticlesInPoly(cHullx, cHully, Np, minDist, useHullVerts = False, rng=None, method = "PD"):
    """
    Parameters
    ----------
    cHullx, cHully : array-like
        Vertices of a (convex) polygon in order. If closed (last==first), the last
        vertex is dropped to make it open (like the MATLAB code).
    Np : int
        Total number of particles desired (including the convex hull vertices).
    minDist : float
        Approximate grid spacing used to place extra particles.
    rng : numpy.random.Generator, optional
        Random generator for reproducibility. If None, uses default_rng().

    Returns
    -------
    newLocs : (Np, 2) ndarray
        Rows are [x, y] for the placed particles: first the newly sampled grid points,
        then the original convex hull vertices (order preserved).
    """
    if rng is None:
        rng = np.random.default_rng()

    # Convert to 1D numpy arrays
    cHullx = np.asarray(cHullx).reshape(-1)
    cHully = np.asarray(cHully).reshape(-1)

    if cHullx.size != cHully.size:
        raise ValueError("cHullx and cHully must have the same length.")

    # Ensure open polygon (drop last vertex if it repeats the first)
    if cHullx.size >= 2 and (cHullx[-1] == cHullx[0]) and (cHully[-1] == cHully[0]):
        cHullx = cHullx[:-1]
        cHully = cHully[:-1]

    M = cHullx.size

    if useHullVerts == True:
        K = int(Np) - M  # number of points to draw inside the hull
    else:
        K = int(Np)

    if K < 0:
        raise ValueError("Np must be at least the number of convex hull vertices.")
    
    # initialize 
    newLocs =  np.array([])

    if method == "grid":
        #############################################
        # Grid method
        #############################################
        gridShift = 0
        placedParticles = False
        while (placedParticles == False) and (gridShift < minDist):
            # Bounding box
            xmin, xmax = float(np.min(cHullx)) - gridShift, float(np.max(cHullx)) 
            ymin, ymax = float(np.min(cHully)) - gridShift, float(np.max(cHully))

            # midpoints
            xmid = xmin + (xmax - xmin)/2
            ymid = ymin + (ymax - ymin)/2

            # Create grid
            Nx = int(np.floor((xmax - xmin) / minDist)) + 1
            Ny = int(np.floor((ymax - ymin) / minDist)) + 1

            xlin = createRangeWithMidpoint(xmin, xmax, xmid, minDist)
            ylin = createRangeWithMidpoint(ymin, ymax, ymid, minDist)
            xGrid, yGrid = np.meshgrid(xlin, ylin, indexing="xy")

            xyGrid = np.column_stack([xGrid.ravel(), yGrid.ravel()])
            NxyGrid = xyGrid.shape[0]
            if NxyGrid < Np:
                raise ValueError("Grid is too coarse to host Np points; increase density (reduce minDist).")

            # check which points are in polygon
            polygon = shapely.Polygon( np.column_stack((cHullx, cHully) ) )
            admissibleLocs = np.zeros(NxyGrid)
            for ixy in range(NxyGrid):
                admissibleLocs[ixy] = polygon.buffer(1E-2).contains(shapely.Point(xyGrid[ixy,0], xyGrid[ixy,1]))

            # Exclude convex hull vertices from admissible locations
            if useHullVerts == True:
                hullPts = np.column_stack([cHullx, cHully])
                cHullRowMask = np.array([(row == hullPts).all(axis=1).any() for row in xyGrid])
                admissibleLocs[cHullRowMask == True] = False

            admissibleIdx = np.flatnonzero(admissibleLocs)


            # Draw K indices
            if len(admissibleIdx) >= K:

                if K > 0:
                    if admissibleIdx.size == 1 and K == 1:
                        LocIdx = admissibleIdx
                    else:
                        LocIdx = rng.choice(admissibleIdx, size=K, replace=False)
                    
                    # get new points
                    new_points = xyGrid[LocIdx, :]

                    # add to existing points from convex hull (if used)
                    if useHullVerts == True:
                        newLocs = np.vstack([new_points, np.column_stack([cHullx, cHully])])
                    else:
                        newLocs = new_points
                else:
                    # No extra points—just return the hull vertices
                    newLocs = np.column_stack([cHullx, cHully])

                placedParticles = True
            else:
                gridShift += 1

    elif method == "PD":
        #############################################
        # Poisson disk method
        #############################################
        xmin, xmax = float(np.min(cHullx)), float(np.max(cHullx)) 
        ymin, ymax = float(np.min(cHully)), float(np.max(cHully))

        maxAttempts = 100
        placedParticles = False
        nAttempts = 0
        while placedParticles == False and nAttempts <= maxAttempts:
            nAttempts += 1

            # Poisson disk sampling
            engine = sc.stats.qmc.PoissonDisk(
                    d=2,
                    radius=minDist,
                    l_bounds=(0,0),
                    u_bounds=(-xmin+xmax,-ymin+ymax) # shift to avoid bug in PoissonDisk implementation
            )

            candidatePoints = engine.fill_space() + np.array([xmin, ymin])  # shift back

            nCands = np.shape(candidatePoints)[0]

            # check which points are in polygon
            polygon = shapely.Polygon( np.column_stack((cHullx, cHully) ) )
            admissibleLocs = np.zeros( nCands )
            for ixy in range(nCands):
                admissibleLocs[ixy] = polygon.buffer(1E-2).contains(shapely.Point(candidatePoints[ixy,0], candidatePoints[ixy,1]))

            admissibleIdx = np.flatnonzero(admissibleLocs)

            # Draw K indices
            if len(admissibleIdx) >= K:
                if admissibleIdx.size == 1 and K == 1:
                    LocIdx = admissibleIdx
                else:
                    LocIdx = rng.choice(admissibleIdx, size=K, replace=False)

                # get new points
                newLocs = candidatePoints[LocIdx, :]
                placedParticles = True


    # if placedParticles == False:
    #     raise ValueError("Could not place particles...")
    
    return newLocs

def KirkwoodRisemanSolve(particleLocs, eta, zeta, particleR, h, dragCoeff):
    
    # number of particles
    Np = particleLocs.shape[0]

    # rhs vector for translation 
    rhsVecXT, rhsVecYT = np.zeros((2*Np,1)), np.zeros((2*Np,1))
    rhsVecXT[0::2] = -dragCoeff 
    rhsVecYT[1::2] = -dragCoeff

    # compute Oseen tensors of particle-particle interactions
    OseenTCart = computeRPYTensor(particleLocs, eta, zeta, particleR, h)

    # assemble interaction matrix
    cMat = CreateMat(OseenTCart, dragCoeff, len(particleLocs[:]))

    # solve linear system
    FvecX = np.linalg.solve( cMat, rhsVecXT)
    FvecY = np.linalg.solve( cMat, rhsVecYT)

    # compute total force on particles due to translation
    Fxx = np.sum(-FvecX[0::2]) 
    Fxy = np.sum(-FvecX[1::2])
    Fyy = np.sum(-FvecY[1::2]) 

    # normalized drag coefficients
    Xixx = Fxx/dragCoeff
    Xixy = Fxy/dragCoeff
    Xiyy = Fyy/dragCoeff

    return Xixx, Xixy, Xiyy


def compHullSampling(particleLocs, particleR, bondLength, eta, zeta, h, dragCoeff, ell, method, methodFactor):

    # number of particles
    Np = np.shape(particleLocs)[1]

    # find convex/concave hull or buffer
    if method == 0:
        cHull = shapely.convex_hull(shapely.MultiPoint(particleLocs.transpose()))
        cHull = shapely.buffer( cHull, methodFactor, quad_segs=8)
        useHullVerts = False

    elif method == 1:
        cHull = shapely.concave_hull(shapely.MultiPoint(particleLocs.transpose()), ratio = methodFactor)
        useHullVerts = False
    elif method == 2:
        cHull = shapely.buffer(shapely.MultiPoint(particleLocs.transpose()), max(methodFactor, (2*particleR + bondLength)))

        useHullVerts = False

    # remove collinear points from polygon
    cHull = cHull.simplify(0) 

    # sample uniformly in approximate geometry
    cHullLocs = np.asarray(cHull.exterior.coords, dtype=float)
    riSample = placeParticlesInPoly(cHullLocs[:,0],cHullLocs[:,1], Np, 2*particleR + bondLength, useHullVerts=useHullVerts, method="PD")

    # if poisson sampling did not succeed, use grid method
    if riSample.size == 0:
        print("trying grid...")
        riSample = placeParticlesInPoly(cHullLocs[:,0],cHullLocs[:,1], Np, 2*particleR + bondLength, useHullVerts=useHullVerts, method="grid")

    # check whether sampling succeeded
    if riSample.size == 0:
        raise ValueError("Could not place particles...")

    # compute geometric mean of interparticle distance 
    Rij = np.zeros( int(Np * (Np-1)/2) )
    cnt = 0
    for ir in range(Np):
        for jr in range(0, ir):
            Rij[cnt] = np.linalg.norm( riSample[ir,:] - riSample[jr,:] )
            cnt += 1
    geomMRij = sc.stats.gmean(Rij)


    # apply KR theory to random sample
    XixxApprox, XixyApprox, XiyyApprox = KirkwoodRisemanSolve(riSample, eta, zeta, particleR, h, dragCoeff)

    return XixxApprox, XixyApprox, XiyyApprox, geomMRij
