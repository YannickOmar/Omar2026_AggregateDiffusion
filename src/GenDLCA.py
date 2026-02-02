import numpy as np
import matplotlib.pyplot as plt
import os 
import graph_tool.all as gt
import random
from collections import deque
# Algorithm (with uniform probability) presented in Kolb, Max, Robert Botet, and Rmi Jullien. "Scaling of kinetically growing clusters." Physical Review Letters 51.13 (1983): 1123.

#### directions: up: 0, left: 1, down: 2, right: 3

def unwrap_periodic_by_propagation(coords, L, edges=None, start=0):
    """
    Unwrap 2D periodic lattice coordinates to a non-periodic lattice so that
    every edge connects nodes at unit Manhattan distance (±1 along one axis).

    Parameters
    ----------
    coords : (N, 2) int ndarray
        Integer lattice coordinates in [0, L-1] x [0, L-1].
    L : int
        Lattice size (period) along each axis.
    edges : iterable[(u, v)], optional
        Undirected edge list over vertex indices. If None, edges are inferred
        from periodic adjacency in `coords`.
    start : int
        Root vertex to start the BFS propagation.

    Returns
    -------
    unwrapped : (N, 2) int ndarray
        Unwrapped coordinates on Z^2 (no periodic wrap).
    """
    coords = np.asarray(coords, dtype=int)
    N = coords.shape[0]

    def min_image_delta(du):
        # map delta to the minimal image in [-floor(L/2), floor(L/2)]
        if du > L // 2:
            du -= L
        elif du < -L // 2:
            du += L
        return int(du)

    # If no edge list is provided, infer edges from periodic nearest-neighbor relations.
    if edges is None:
        inferred = []
        for i in range(N):
            xi, yi = coords[i]
            for j in range(i + 1, N):
                xj, yj = coords[j]
                dx = min_image_delta(xj - xi)
                dy = min_image_delta(yj - yi)
                if abs(dx) + abs(dy) == 1:   # von Neumann neighbors with wrap
                    inferred.append((i, j))
        edges = inferred

    # Build adjacency
    nbrs = [[] for _ in range(N)]
    for u, v in edges:
        nbrs[u].append(v)
        nbrs[v].append(u)

    # BFS propagation
    unwrapped = np.zeros_like(coords, dtype=int)
    visited = np.zeros(N, dtype=bool)
    q = deque([start])
    visited[start] = True
    unwrapped[start] = coords[start]

    while q:
        u = q.popleft()
        xu, yu = coords[u]
        for v in nbrs[u]:
            xv, yv = coords[v]
            dx = min_image_delta(xv - xu)
            dy = min_image_delta(yv - yu)

            if not visited[v]:
                unwrapped[v, 0] = unwrapped[u, 0] + dx
                unwrapped[v, 1] = unwrapped[u, 1] + dy
                visited[v] = True
                q.append(v)
            else:
                # Consistency check for cycles: correct by multiples of L if needed
                expected = unwrapped[u] + np.array([dx, dy], dtype=int)
                corr = expected - unwrapped[v]
                # If a cycle introduces a net L-shift, allow it; otherwise it's inconsistent.
                if (corr[0] % L != 0) or (corr[1] % L != 0):
                    raise ValueError("Inconsistent wrapping detected: graph/coords mismatch.")
                # If corr is a nonzero multiple of L, shift v (and everything connected to v that’s already placed)
                if corr[0] or corr[1]:
                    # Shift all already-placed nodes that are connected to v through already-placed edges.
                    # For DLCA clusters (tree-like), this rarely triggers; keep it simple:
                    unwrapped[v] += corr

    # If there are any unvisited nodes (shouldn’t happen for your final single cluster),
    # unwrap each remaining component independently:
    if not np.all(visited):
        for root in np.where(~visited)[0]:
            q = deque([root])
            visited[root] = True
            unwrapped[root] = coords[root]
            while q:
                u = q.popleft()
                xu, yu = coords[u]
                for v in nbrs[u]:
                    xv, yv = coords[v]
                    dx = min_image_delta(xv - xu)
                    dy = min_image_delta(yv - yu)
                    if not visited[v]:
                        unwrapped[v] = unwrapped[u] + np.array([dx, dy])
                        visited[v] = True
                        q.append(v)

    # Final assertion: every edge is a unit step in the unwrapped lattice.
    for u, v in edges:
        d = np.abs(unwrapped[v] - unwrapped[u]).sum()
        assert d == 1, f"Edge ({u},{v}) not unit after unwrapping (|Δ|₁={d})."

    return unwrapped

N = 20      # number of particles in cluster
nSample = 5 # number of samples of DLCA with N particles

latticeSizeFactor = 3   # system size is latticeSizeFactor * sqrt(N)

# filename for output
outDir = "data"
outFname = "DLCA_N" + str(N) + ".txt"

############## initialization ###################

# create output directory 
if os.path.isdir(outDir)==False:
    os.mkdir(outDir)

# relative position vector array 
latticeVectors = np.array([[0,1], [-1,0], [0,-1], [1,0]])

# output data
allSamples = np.zeros((2*nSample, N))

# number of lattice sites per direction
Nlattice = np.floor(latticeSizeFactor * np.sqrt(N)).astype(int)

############## run MCMC ###################
print("Starting sampling...")

for iSample in range(0,nSample):

    # initialize graph
    g = gt.Graph(directed=False)

    # add vertices
    g.add_vertex(N)

    # generate unique positions 
    uniquePosSet = set() # using a set ensure unqiueness
    while len(uniquePosSet) < N:
        col = tuple(np.random.randint(0, Nlattice, size=2))  # Generate a random column
        uniquePosSet.add(col) 
    initLocs = np.array(list(uniquePosSet))

    # initialize position vector (can have at most N+1 vertices) 
    posProp = g.new_vertex_property("vector<int>") 
    for iV in range(0,N):
        posProp[g.vertex(iV)] = np.round(initLocs[iV]).astype(int)

    # check for initial binding
    for iVOut in range(0,N):
        outPos = posProp[iVOut]
        for iVIn in range(0,N):
            inPos = posProp[iVIn]
            if (((np.abs(outPos[0] - inPos[0]) == 1 or np.abs(outPos[0] + Nlattice - inPos[0]) == 1 or np.abs(outPos[0] - Nlattice - inPos[0]) == 1) and np.abs(outPos[1] - inPos[1]) == 0) or 
                ((np.abs(outPos[1] - inPos[1]) == 1 or np.abs(outPos[1] + Nlattice - inPos[1]) == 1 or np.abs(outPos[1] - Nlattice - inPos[1]) == 1) and np.abs(outPos[0] - inPos[0]) == 0)):
                g.add_edge(iVOut, iVIn)

    # number of clusters
    nClusters = N

    # just for the loop to work properly, we do this once beforehand
    comp, _ = gt.label_components(g)

    # DLCA algorithm
    while nClusters > 1:

        # pick random cluster 
        iCluster = random.randint(0,nClusters-1)

        # pick move 
        moveDirection = random.randint(0,3)

        # get particles inside and outside of cluster
        vertex2Sub = comp.get_array()
        iniCluster = np.where( vertex2Sub == iCluster )[0]
        NotIniCluster = np.where( vertex2Sub != iCluster )[0]
        
        # update position vector
        for iV in iniCluster:
            posProp[iV][0] = (posProp[iV][0] + latticeVectors[moveDirection][0]) % Nlattice
            posProp[iV][1] = (posProp[iV][1] + latticeVectors[moveDirection][1]) % Nlattice

        # check for binding 
        for iVOut in range(0,len(NotIniCluster)):
            outPos = posProp[NotIniCluster[iVOut]]
            for iVIn in range(0,len(iniCluster)):
                inPos = posProp[iniCluster[iVIn]]
                if (((np.abs(outPos[0] - inPos[0]) == 1 or np.abs(outPos[0] + Nlattice - inPos[0]) == 1 or np.abs(outPos[0] - Nlattice - inPos[0]) == 1) and 
                     (np.abs(outPos[1] - inPos[1]) == 0 or np.abs(outPos[1] + Nlattice - inPos[1]) == 0 or np.abs(outPos[1] - Nlattice - inPos[1]) == 0)) or 
                    ((np.abs(outPos[1] - inPos[1]) == 1 or np.abs(outPos[1] + Nlattice - inPos[1]) == 1 or np.abs(outPos[1] - Nlattice - inPos[1]) == 1) and 
                     (np.abs(outPos[0] - inPos[0]) == 0 or np.abs(outPos[0] + Nlattice - inPos[0]) == 0 or np.abs(outPos[0] - Nlattice - inPos[0]) == 0))):
                    g.add_edge(NotIniCluster[iVOut], iniCluster[iVIn])

        # recompute compnents and update number of clusters
        comp, ghist = gt.label_components(g)
        nClusters = len(ghist)

    # store current positions 
    cLocs = posProp.get_2d_array()
    unwrappedLocs = unwrap_periodic_by_propagation(cLocs.T, Nlattice, g.get_edges()).T
    allSamples[(2*iSample):(2*(iSample+1))] = unwrappedLocs - np.reshape(unwrappedLocs[:,0], (-1,1))

    assert(len(set(map(tuple, cLocs.T))) == cLocs.shape[1])

# write to file
np.savetxt(outDir + "/" + outFname, allSamples)


