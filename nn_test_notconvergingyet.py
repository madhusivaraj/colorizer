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
    if label == 0:
        return z
    elif label == 1:
        return 1/(1+np.exp(-z))

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
            #loss /= len(f)
            return loss

def lossPrime(z, f, label):
    if label == 0: # L2 norm
        return 2*(z-f)

def sigmaPrime(z, label):
    # 0 = no act fcn
    # 1 = sigmoid
    if label == 0:
        return 1
    if label == 1:
        return z*(1-z)

def stepSize():
    return 1

def backProp(nodes, deltas, weights, f):
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
                    delta += deltas[layer+1][k] * sigmaPrime(np.dot(weights[layer][k], nodes[layer]), actL[layer]) * weights[layer][k][j]
                deltas[layer][j] = delta
    # update weights
    for i in range(len(weights)-1): # for each layer, except final
        for j in range(len(weights[i])): # for each group of weights in layer
            for k in range(len(weights[i][j])): # for each weight in group
                step = stepSize() * deltas[i+1][j] * sigmaPrime(np.dot(weights[i][j], nodes[i]), actL[layer]) * nodes[i][k]
                weights[i][j][k] -= step
    return weights
    


##########################################################################
##########################################################################
# Driver Script
numL = 4
npL = [2, 10, 10, 1] # includes biases
biasL = [1, 1, 1, 0] # either 1 or 0 for each layer
actL = [0, 1, 1, 1] # label indicating act fcn leading into that layer; 1st is meaningless
[nodes, deltas, weights] = nnInitialize(numL, npL)
lossL = 0
#print(nodes)
#print(deltas)
#print(weights)
#X = [[1, 0], [0,1]]
#F = [[1, 1], [0, 0]]
X=[[1]]
F=[[1]]
iterations = int(1e3)
lossList = np.zeros(iterations)
for i in range(iterations):
    #r = np.random.randint(len(X))
    r = 0
    x = X[r]
    f = F[r]
    nodes = forwardProp(x, nodes, weights, biasL, actL)
    lossList[i] = loss(nodes, f, 0)
    weights = backProp(nodes, deltas, weights, f)
fig, ax = plt.subplots()
ax.plot(lossList)
ax.grid()
#fig.savefig("test.png")
plt.show()

print('Done')

