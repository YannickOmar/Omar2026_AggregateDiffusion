import numpy as np
import matplotlib.pyplot as plt
import os 
import graph_tool.all as gt
import random
# Algorhtm presented in E J Janse van Rensburg and N Madras 1997 J. Phys. A: Math. Gen. 30 8035

#### directions: up: 0, left: 1, down: 2, right: 3

N = 20      # number of edges in LA
nSample = 5 # number of samples of LA with N edges

# filename for output
outDir = "data"
outFname = "LA_N" + str(N) + ".txt"

nEquilibrate = 2*N        # number of accepted equilibration steps
nWaitSample = N      # number of accepted steps before new sample is drawn 
maxAttempts = 5*N      # maximum number of attempts to generate new sample

############## initialization ###################

# create output directory 
if os.path.isdir(outDir)==False:
    os.mkdir(outDir)

# initialize graph
g = gt.Graph(directed=False)

# add edge list 
edgeSources = np.linspace(0,N-1,N)
edgesTargets = np.linspace(1,N,N)
edgesList = np.vstack([edgeSources, edgesTargets])
g.add_edge_list(edgesList.transpose())

# initialize position vector (can have at most N+1 vertices) 
posProp = g.new_vertex_property("vector<int>") 
for iV in range(0,N+1):
    posProp[g.vertex(iV)] = [iV, 0]

# relative position vector array 
latticeVectors = np.array([[0,1], [-1,0], [0,-1], [1,0]])

# # output data
allSamples = np.zeros((2*nSample, N+1))

# counters
iSample = 0
iSinceLastSample = 0
iAttempts = 0

############## run MCMC ###################
print("Starting sampling...")

while (iSample < nSample) and iAttempts < maxAttempts:

    # get all edges
    cAllEdges = g.get_edges()

    # choose edge randomly
    cEdgeIdx = random.randint(0, N-1)

    # neighboring vertices
    cEdgeSource = cAllEdges[cEdgeIdx,0]
    cEdgeTarget = cAllEdges[cEdgeIdx,1]

    # source and target vertex object
    sourceVertex = g.vertex(cEdgeSource)
    targetVertex = g.vertex(cEdgeTarget)

    # edges associated with source and target
    sourceEdges = g.get_all_edges(sourceVertex)
    targetEdges = g.get_all_edges(targetVertex)

    if sourceEdges.shape[0] == 1:
        isLeaf = True
        keepVertex = targetVertex
        removeVertex = sourceVertex
    elif targetEdges.shape[0] == 1:
        isLeaf = True
        keepVertex = sourceVertex
        removeVertex = targetVertex
    else: 
        isLeaf = False

    moveRejected = False 
    
    # copy of current graph in case we reject the new graph
    prevG = gt.Graph(g)
    prevPosProp = prevG.copy_property(posProp) 

    if isLeaf == True:
        # perform leaf move

        # remove edge
        g.remove_edge(g.edge(sourceVertex, targetVertex))

        # remove vertex from graph
        g.remove_vertex(removeVertex)

        # pick random vertex
        uVertex = random.randint(0,g.num_vertices()-1)

        # pick neighboring vertex 
        vVertexIdx = random.randint(0,3)

        # location of uVertex
        uPos = np.array(posProp[uVertex])
        
        # compute position vector of vertex 
        vPos = uPos + latticeVectors[vVertexIdx]
        vPos = np.round(vPos).astype(int)

        # check whether position is already occupied (potential contact edge)
        isOccupied = False
        for iVertex in range(0, g.num_vertices()):

            cPos = np.array(posProp[iVertex])

            if cPos[0] == vPos[0] and cPos[1] == vPos[1]:
                isOccupied = True 

                #  check whether edge exists already 
                vNeighbors = g.get_all_neighbors(iVertex)
                if any(np.isin([uVertex], vNeighbors)) or (any(np.isin([uVertex], vNeighbors)) == False and random.randint(0,1) == 0):
                    moveRejected = True
                else: # edge does not exist yet (add with probability 1/2 (see if (...)))
                    g.add_edge(uVertex, iVertex)

                break

        # perimeter edge: add new vertex and edge
        if isOccupied == False:
            newVertex = g.add_vertex()
            posProp[newVertex] = vPos
            g.add_edge(uVertex, newVertex)

    # not a leaf 
    else:
        ### check for subanimal vs cycle move
        
        # remove edge
        g.remove_edge(g.edge(sourceVertex, targetVertex))

        _, ghist = gt.label_components(g)

        if len(ghist) != 1: # edge removal creates two subanimals 
            # perform sub-animal move

            # find smaller subgraph
            smallSub = np.argmin(ghist)
            largeSub = 1 - smallSub

            # label components
            compLabels,_ = gt.label_components(g)
            compLabelArray = compLabels.a

            # find vertices in small and largest subgraph
            smallSubVertices = np.where(compLabelArray == smallSub)[0]
            largeSubVertices = np.where(compLabelArray == largeSub)[0]

            # pick random vertices on smallest and largest subgraphs
            attachVertexSmall = smallSubVertices[random.randint(0,len(smallSubVertices)-1)]
            attachVertexLarge = largeSubVertices[random.randint(0,len(largeSubVertices)-1)]

            # location of attachment point of the smaller subanimal
            attachVertexSmallNewLoc = posProp[attachVertexLarge] + latticeVectors[random.randint(0,3)]
            attachVertexSmallNewLoc = np.round(attachVertexSmallNewLoc).astype(int)

            # pick rotation angle
            rotAngle = 1/2 * np.pi * random.randint(0,3)

            # set up rotation matrix
            Rrot = np.matrix( [[np.cos(rotAngle), -np.sin(rotAngle)], [np.sin(rotAngle), np.cos(rotAngle)]] )

            # get positions of smaller subgraph
            smallSubLocs = np.zeros((2,len(smallSubVertices)))
            for iV in range(0,len(smallSubVertices)):
                relPos = np.reshape(np.array(posProp[smallSubVertices[iV]]) - np.array(posProp[attachVertexSmall]), (2,1))
                rotPos = Rrot * relPos
                smallSubLocs[:,iV] = np.reshape(np.round(rotPos).astype(int) + np.reshape(attachVertexSmallNewLoc, (2,1)), (2,))

            # check for overlaps
            noOverlap = True
            for iVSmall in range(0, len(smallSubVertices)):
                smallPos = np.round(smallSubLocs[:,iVSmall]).astype(int)
                # print("\n smallPos: {:d}, {:d}".format(smallPos[0], smallPos[1]))
                for iVLarge in largeSubVertices:
                    largePos = np.array(posProp[iVLarge])
                    # print("largePos: {:d}, {:d}".format(largePos[0], largePos[1]))
                    if smallPos[0] == largePos[0] and smallPos[1] == largePos[1]:
                        noOverlap = False 
                        break
                if noOverlap == False:
                    break
   
            if noOverlap == True: # accept move
                # add edge
                g.add_edge(attachVertexSmall,attachVertexLarge)

                # update positions of small subanimal vertices
                for iV in range(0,len(smallSubVertices)):
                    posProp[smallSubVertices[iV]] = smallSubLocs[:,iV]

            else: # reject move and add edge back in 
                moveRejected = True

        else:  # edge is part of cycle

            # pick random vertex
            uVertex = random.randint(0,g.num_vertices()-1)

            # pick neighboring vertex 
            vVertexIdx = random.randint(0,3)

            # location of uVertex
            uPos = np.array(posProp[uVertex])
            
            # compute position vector of vertex 
            vPos = uPos + latticeVectors[vVertexIdx]
            vPos = np.round(vPos).astype(int)

            # check whether position is already occupied (potential contact edge)
            isOccupied = False
            for iVertex in range(0, g.num_vertices()):
                cPos = np.array(posProp[iVertex])

                if cPos[0] == vPos[0] and cPos[1] == vPos[1]:
                    isOccupied = True 

                    #  check whether edge exists already 
                    vNeighbors = g.get_all_neighbors(iVertex)
                    if any(np.isin([uVertex], vNeighbors)):
                        # edge exists already -> reject move 
                        moveRejected = True
                    else: # edge does not exist yet (add with probability 1/2 (see if (...)))
                        g.add_edge(uVertex, iVertex)

                    break

            # perimeter edge: add new vertex and edge
            if isOccupied == False:
                newVertex = g.add_vertex()
                posProp[newVertex] = vPos
                g.add_edge(uVertex, newVertex)

    if moveRejected == True:
        g = gt.Graph(prevG)
        posProp = g.copy_property(prevPosProp)
        iAttempts += 1
    else: 
        iAttempts = 0
        if ((iSinceLastSample == nWaitSample-1) and (iSample != 0)) or ((iSinceLastSample == nEquilibrate-1) and (iSample == 0)):

            # get current positions and store 
            currentLocs = posProp.get_2d_array()
            for iV in range(0,N+1):
                if iV < currentLocs.shape[1]:
                    allSamples[(2*iSample):(2*(iSample+1)), iV] = currentLocs[:,iV] - currentLocs[:,0]
                else:
                    allSamples[(2*iSample):(2*(iSample+1)), iV] = np.array([np.nan, np.nan])

            # set counters
            iSinceLastSample = 0
            iSample += 1

        else:
            iSinceLastSample += 1

if iAttempts == maxAttempts: 
    raise ValueError("Couldn't generate new sample")

# write to file (TODO: should avoid writing NaNs to file) 
np.savetxt(outDir + "/" + outFname, allSamples)


