import numpy as np
import matplotlib.pyplot as plt

def nnInitialize(numL, npL):
    # numL = number of layers (whole number)
    # npL = nodes per layer (vector of whole numbers, numL in len)
    # actL = activation fcn of each layer (vector of tags, numL in len)
    nodes = []
    deltas = []
    for i in range(numL):
        nodes.append([])
        deltas.append([])
        layerSize = npL[i]
        for j in range(layerSize):
            nodes[i].append(0)
            deltas[i].append(0)
    weights = []
    for i in range(numL-1):
        weights.append([])
        wpGroup = len(nodes[i]) # weights per weight group (for each node)
        numGroups = len(nodes[i+1])# num of weight groups (num nodes in next layer)
        VarXav = 1/(0.5*(len(nodes[i])+len(nodes[i+1]))) # Xavier initialization variance
        for j in range(numGroups):
            weights[i].append(np.random.normal(0, VarXav, wpGroup))
    return [nodes, deltas, weights]

def sigma(z, label):
    # activation fcn of a given node
    # label indicates which fcn to use
    # 0 = no act fcn (output = input)
    # 1 = sigmoid
    # 2 = ReLU
    # 3 = leaky ReLU
    if label == 0:
        return z
    elif label == 1:
        return 1/(1+np.exp(-z))
    elif label == 2:
        return max(0,z)
    elif label == 3:
        if z > 0:
            return z
        else:
            return z*0.001
    elif label == 4:
        return np.tanh(z)

def forwardProp(x, nodes, weights, biasL, actL):
    # x = input vector
    # nodes = node structure of nn
    # weights = weights of nn
    for i in range(len(nodes)):
        if i == 0: # for layer 0, input x values
            if len(nodes[i])-len(x) != biasL[i]: # checking biasing correctly
                print('ERROR: Check input layer bias.')
            for j in range(len(x)): # fill input layer with x values
                nodes[i][j] = x[j]
        else:
            for j in range(len(nodes[i])):
                nodes[i][j] = sigma(np.dot(weights[i-1][j], nodes[i-1]), actL[i])
        
        if biasL[i] == 1: # if layer has a bias
            nodes[i][len(nodes[i])-1] = 1 #bias for that layer
    return nodes

def loss(nodes, f, label):
    # calculates loss after forwardProp by comparing
    # actual output to desired output
    loss = 0
    if label == 0: # L2 norm
        for i in range(len(f)):
            loss += (nodes[len(nodes)-1][i]-f[i])**2
            loss /= len(f)
            return loss

def lossPrime(z, f, label):
    if label == 0: # L2 norm
        return 2*(z-f)

def sigmaPrime(z, label):
    # 0 = no act fcn
    # 1 = sigmoid
    # 2 = ReLU
    # 3 = leaky ReLU
    # 4 = tanh
    if label == 0:
        return 1
    if label == 1:
        return z*(1-z)
    if label == 2:
        if z > 0:
            return z
        else:
            return 0
    if label == 3:
        if z > 0:
            return z
        else:
            return z*0.001
    if label == 4:
        return 1-np.tanh(z)**2
        
def stepSize():
    return 1

def backProp(nodes, deltas, weights, f):
    # copy "weights"
    weights2 = []
    for i in range(len(weights)): # each layer
        weights2.append([])
        for j in range(len(weights[i])): # each group in layer
            weights2[i].append([])
            for k in range(len(weights[i][j])): # each weight in group
                weights2[i][j].append(weights[i][j][k])
    # calculate deltas
    for i in range(len(nodes)): # for each layer
        layer = len(nodes)-1-i # count backwards thru layers
        if layer+1 == len(nodes): # if it's the final layer
            for j in range(len(nodes[layer])): # for each node in layer
                deltas[layer][j] = lossPrime(nodes[layer][j], f[j], lossL)
        elif layer != 0: # neglect layer 0
            for j in range(len(nodes[layer])): # for each node in layer
                delta = 0
                for k in range(len(nodes[layer+1])): # for each node in next layer
                    delta += deltas[layer+1][k] * sigmaPrime(np.dot(weights2[layer][k], nodes[layer]), actL[layer]) * weights2[layer][k][j]
                deltas[layer][j] = delta
    # update weights
    for i in range(len(weights2)-1): # for each layer, except final
        for j in range(len(weights2[i])): # for each group of weights in layer
            for k in range(len(weights2[i][j])): # for each weight in group
                step = stepSize() * deltas[i+1][j] * sigmaPrime(np.dot(weights2[i][j], nodes[i]), actL[layer]) * nodes[i][k]
                #if i ==0 and j == 0 and k == 0:
                    #print('step',step)
                weights2[i][j][k] -= step
    return weights2

def plotWeights(weightList, npL):
    # calc num of weights
    numWeights = 0
    for i in range(len(npL)-1):
        numWeights += npL[i+1]*npL[i]
        #print('numweights',numWeights)
    # make list with one row per weight
    weightTrajectories = []
    for i in range(numWeights):
        weightTrajectories.append([])
        for j in range(len(weightList)):
            weightTrajectories[i].append(0)
    #print('len w traj',len(weightTrajectories))
    # fill with weights
    wNum = 0
    for i in range(len(weights)): # each layer
        for j in range(len(weights[i])): # each w group in layer
            for k in range(len(weights[i][j])): # each w in group
                for l in range(len(weightList)): # for each iteration of that w
                    weightTrajectories[wNum][l] = weightList[l][i][j][k]
                plt.plot(weightTrajectories[wNum])
                wNum += 1
    #print('wNum',wNum)
    plt.show()
    return weightTrajectories
                    
    


##########################################################################
##########################################################################
# Driver Script
numL = 4
npL = [3, 10, 3, 2] # includes biases
biasL = [1, 1, 1, 0] # either 1 or 0 for each layer; last always 0
actL = [0, 1, 1, 1] # label indicating act fcn leading into that layer; 1st is meaningless
[nodes, deltas, weights] = nnInitialize(numL, npL)
lossL = 0
#print(nodes)
#print(deltas)
    #print(weights)
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
F = [[0, 1], [1, 0], [1, 1], [0, 0]]
#F = [ [0,0], [0,0], [0,0], [0,0] ]
#X=[[1]]
#F=[[1]]
iterations = int(1e3)
lossList = np.zeros(iterations)
#weightList = [0]*iterations#np.zeros(iterations)
weightList = []
for i in range(iterations):
    r = np.random.randint(len(X))
    #r = 0
    x = X[r]
    f = F[r]
    nodes = forwardProp(x, nodes, weights, biasL, actL)
    lossList[i] = loss(nodes, f, 0)
    weights = backProp(nodes, deltas, weights, f)
    #weightList[i] = weights
    weightList.append(weights)
    #print('w',weights[0][0])
    #print('wlist',weightList)
fig, ax = plt.subplots()
ax.plot(lossList)
#ax.plot(weightList)
ax.grid()
fig.savefig("test.png")
plt.show()

#fig, ax = plt.subplots()
#ax.plot(weightList)
#ax.grid()
#fig.savefig("test.png")
#plt.show()

#print('before',weightList)
weightTrajectories = plotWeights(weightList, npL)
#print('after',weightList)
#print(weightTrajectories[0])
#print(weightTrajectories[1])
print('Done')


