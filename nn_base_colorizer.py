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
    elif label == 1: # log loss
        for i in range(len(f)):
            loss += ( -f[i]*np.log(nodes[len(nodes)-1][i]) - (1-f[i])*np.log(1-nodes[len(nodes)-1][i]))
            loss /= len(f)
            return loss

def lossPrime(z, f, label):
    if label == 0: # L2 norm
        return 2*(z-f)
    if label == 1: # log loss
        return (z-f)*z
        
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
                deltas[layer][j] = lossPrime(nodes[layer][j], f[j], lossL) # calculate delta
        elif layer != 0: # if it's not the final layer (while neglecting layer 0)
            for j in range(len(nodes[layer])): # for each node in layer
                delta = 0
                for k in range(len(nodes[layer+1])): # for each node in next layer
                    delta += deltas[layer+1][k] * sigmaPrime(np.dot(weights2[layer][k], nodes[layer]), actL[layer]) * weights2[layer][k][j]
                deltas[layer][j] = delta # calculate delta
    # update weights
    for i in range(len(weights2)-1): # for each layer, except final
        for j in range(len(weights2[i])): # for each group of weights in layer
            for k in range(len(weights2[i][j])): # for each weight in group
                step = stepSize() * deltas[i+1][j] * sigmaPrime(np.dot(weights2[i][j], nodes[i]), actL[layer]) * nodes[i][k] # calculate step
                weights2[i][j][k] -= step # update weight
    return weights2 # return updated weights

def plotWeights(weightList, npL):
    # calc num of weights
    numWeights = 0
    for i in range(len(npL)-1):
        numWeights += npL[i+1]*npL[i]
    # make list with one row per weight
    weightTrajectories = []
    for i in range(numWeights):
        weightTrajectories.append([])
        for j in range(len(weightList)):
            weightTrajectories[i].append(0)
    # fill with weights
    wNum = 0
    for i in range(len(weights)): # each layer
        for j in range(len(weights[i])): # each w group in layer
            for k in range(len(weights[i][j])): # each w in group
                for l in range(len(weightList)): # for each iteration of that w
                    weightTrajectories[wNum][l] = weightList[l][i][j][k]
                plt.plot(weightTrajectories[wNum])
                wNum += 1
    fig.suptitle('weights vs training iteration')
    plt.show()
    return weightTrajectories
                    
    
##########################################################################
##########################################################################

# Driver Script

# initialize network
numL = 3 # number of layers
inLen = 5*5 # length of input vector
outLen = 3 # length of output vector
biasL = [1, 1, 0] # bias for each layer; 1 = layer has bias, 0 = no bias; last always 0
npL = [(inLen+biasL[0]), 5, outLen] # nodes per layer (includes biases)

actL = [0, 4, 1] # label indicating activation fcn type leading into that layer; 1st is meaningless (see "sigma" fcn)
[nodes, deltas, weights] = nnInitialize(numL, npL) # creates node, delta, and weight lists
lossL = 1 # label indicating loss fcn type (see "loss" and "lossPrime" fcns)

# define training data
#X = [[0, 0], [0, 1], [1, 0], [1, 1]] # input training data points
#F = [[0,1], [0,1], [1,0], [1,0]] # output training data points

# train nn using stochastic gradient descent
iterations = int(1e3) # number of training iterations
lossList = np.zeros(iterations) # list of loss value of each iteration
weightList = [] # list of weight values of each iteration
for i in range(iterations):
    r = np.random.randint(len(X)) # choose random training data point index
    x = X[r] # chosen input data point
    f = F[r] # chosen correspoinding output data point
    nodes = forwardProp(x, nodes, weights, biasL, actL) # calculate nn nodes from input
    lossList[i] = loss(nodes, f, lossL) # calculate loss of nn; add to list
    weights = backProp(nodes, deltas, weights, f) # update nn weights
    weightList.append(weights) # add weights to list

# plot loss vs iteration
fig, ax = plt.subplots()
ax.plot(lossList)
ax.grid()
fig.suptitle('loss vs training iteration')
plt.show()

# plot weights vs iteration
weightTrajectories = plotWeights(weightList, npL)

#
print('Done')


