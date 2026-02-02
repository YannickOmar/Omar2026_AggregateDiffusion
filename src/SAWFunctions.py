import numpy as np
import matplotlib.pyplot as plt

def plotSAW(posVec, filename="output.png"):
    
    xCoords = posVec[0, :]
    yCoords = posVec[1, :]
    
    plt.figure(figsize=(8, 6))
    plt.plot(xCoords, yCoords, '-o', markersize=5, color='blue')  # Lines with circles
    plt.scatter(xCoords, yCoords, color='red', s=30)  # Small filled circles at points
    
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.axis('equal')  # Keep aspect ratio equal for proper scaling
    plt.savefig(filename)
    plt.close()


def performMove(moveType, chainSide, N, kSite, prevPosVec, prevStepVec):

    if chainSide == 0:
        iAdd = -1
        chainEnd = 0
        prevStepVec = ( prevStepVec + 2 ) % 4
    else:
        iAdd = 1
        chainEnd = N-1

    # find move set 
    moveSet = findMoveSet(moveType, prevStepVec[int(kSite)])

    # initialize
    cSite = kSite
    newPosVec = prevPosVec.copy()
    newStepVec = prevStepVec.copy()
    isUnique = True

    while cSite != chainEnd: 

        # move to next site
        prevSite = int(cSite)
        cSite = int( cSite + iAdd ) 

        # update step
        newStepVec[prevSite] = ((prevStepVec[prevSite] + moveSet[int(prevStepVec[prevSite])] ) % 4) 

        # update position
        newPosVec[:,cSite] = updPosition(newStepVec[prevSite], newPosVec[:,prevSite])

        # check for uniqueness
        isUnique = np.unique(newPosVec, axis=1).shape[1] == newPosVec.shape[1]
        if isUnique == False:
            newStepVec = prevStepVec
            newPosVec = prevPosVec
            break
        
    if chainSide == 0:
        newStepVec = ( newStepVec + 2 ) % 4

    return isUnique, newPosVec, newStepVec

def updPosition(step, prevPos):

    if step == 0:
        deltaPos = [0, 1]
    elif step == 1:
        deltaPos = [-1, 0]
    elif step == 2:
        deltaPos = [0, -1]
    elif step == 3:
        deltaPos = [1, 0]

    return (prevPos + deltaPos)

def findMoveSet(move, base=0):

    moveSet = np.zeros(4)

    match move:
        case 0: # rotation: 90 degree
            moveSet = np.ones(4)
        case 1: # rotation: -90 degree 
            moveSet = -np.ones(4)
        case 2: # rotation: 180 degrees
            moveSet = 2*np.ones(4)
        case 3: # axis reflection x
            moveSet = [0,2,0,2]
        case 4: # axis reflection y
            moveSet = [2,0,2,0]
        case 5: # diagonal reflection 1
            moveSet = [3,1,-1,-3]
        case 6: # diagonal reflection 2
            moveSet = [1,-1,1,-1]
    return moveSet