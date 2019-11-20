
import imageio
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.cluster import MiniBatchKMeans

class clusters:
    def __init__(self,centers,labels):
        self.centers = centers
        self.labels = labels


class NeuralNet:
    def __init__(self, nLayers, nNodesPerLayer):
        self.nLayers = nLayers #number of hidden layers + 1 (output layer)
        self.nNodesPerLayer = nNodesPerLayer #number of nodes in each layer
        self.inputs = []
        self.outputs = []
        self.nInput = 9
        self.activationFunctions = []
        self.Deltas = []
        self.nColors = 0
        self.errorType = 'logistic'
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def sigmoidPrime(self,x):
        self.sigmoid(x)-(self.sigmoid(x))**2

    def tanhPrime(self,x):
        return 1-np.tanh(x)**2

    def ReLuBound(self,x):
        return max([0,x])

    def ReLuBoundPrime(self,x):
        if x < 0:
            return 0
        else:
            return 1

    def softmax(self,x,sumexp):
        return np.exp(x)/sumexp

    def lossClassifier(self,index):
        -self.expectedOutput(index)*np.log(self.outputs[-1][index]) - \
        (1-self.expectedOutput[index])*np.log(1-self.outputs[-1][index])

    def lossClassifierPrime(self,index):
        self.outputs[-2][index]*(self.outputs[-1][index] - self.expectedOutput[index])


    def initializeNN(self,alpha):
        # initialize a NN with random initial weights <0.1
        # this NN contains 2 hidden layers, one with activation function tanh and one with the sigmoid activation function.
        # the output layer has three nodes with activation function ReLu
        # each hidden layer except for the output will contain a bias node with the value of 1
        # The representation of the NN will be an nNodes(LayerN-1) x nNodes(Layer N). nNodes(0) = nData
        self.alpha = alpha
        adjMatrices = []
        layerSizes = [self.nInput]
        for i in range(self.nLayers):
            layerSizes.append(self.nNodesPerLayer[i])

        for i in range(self.nLayers):
            tmp = np.random.rand(layerSizes[i+1], layerSizes[i])*0.001
            adjMatrices.append(tmp)

        self.adjMatrices = adjMatrices

        # set the activation function identifiers for each layer:
        # the function identifier 0 corresponds to tanh
        # the function identifier 1 corresponds to the sigmoid function
        # the function identifier 2 corresponds to the ReLu function bound to 255

        self.activationFunctions.append(0)  # hidden layer 1 has tanh as its activation funtion
        self.activationFunctions.append(3)  # output layer has has tanh as its activation funtion

        self.nColors = self.nNodesPerLayer[-1]

    def findClusterCenters(self,imOriginal):
        imOriginal = imOriginal[:, :, :3]
        nrows = imOriginal.shape[0]
        ncols = imOriginal.shape[1]
        im = np.reshape(imOriginal, (int(imOriginal.size / 3), 3))
        kmeans = MiniBatchKMeans(n_clusters=self.nColors, init='k-means++', batch_size=10000, n_init=30).fit(im)
        centrs = kmeans.cluster_centers_
        labels = pairwise_distances_argmin(im, kmeans.cluster_centers_)
        labels = np.reshape(labels, (nrows, ncols))
        return clusters(centrs,labels)


    def SimulateNN(self, input, expectedOutput):
        # simulates a one-layer NN with N input nodes (grayscale values for pixels of the image),
        # and outputs 3N values (RGB values for pixels of the image)
        # the activation functions will be the logistic curve, the hyperbolic function, the square and the cube
        self.input = input
        self.expectedOutput = expectedOutput
        self.outputs = []
        self.inputs = []
        for i in range(self.nLayers):
            if i == 0:
                self.inputs.append(np.dot(self.adjMatrices[i], self.input))
            else:
                self.inputs.append(np.dot(self.adjMatrices[i], self.outputs[i-1]))

            if self.activationFunctions[i] == 0:
                self.outputs.append([np.tanh(self.inputs[i][j]) for j in range(self.nNodesPerLayer[i])])
            elif self.activationFunctions[i] == 1:
                self.outputs.append([self.sigmoid(self.inputs[i][j]) for j in range(self.nNodesPerLayer[i])])
            elif self.activationFunctions[i] == 2:
                self.outputs.append([self.ReLuBound(self.inputs[i][j]) for j in range(self.nNodesPerLayer[i])])
            elif self.activationFunctions[i] == 3:
                sumExpInputs = sum([np.exp(self.inputs[i][j]) for j in range(self.nNodesPerLayer[i])])
                self.outputs.append([self.softmax(self.inputs[i][j], sumExpInputs) for j in range(self.nNodesPerLayer[i])])

        if self.errorType == 'sqerror':
            self.error = sum([(self.outputs[-1][i] - expectedOutput[i])**2 for i in range(len(expectedOutput))])
        elif self.errorType == 'logistic':
            self.error = sum(sum([-self.expectedOutput[i]*np.log(self.outputs[-1][i]) -
                              (1-self.expectedOutput)*np.log(1-self.outputs[-1][i])]))

    def errorBackpropagation(self):
        self.Deltas = []
        adjMatrices = self.adjMatrices
        for layer in range(self.nLayers):
            self.Deltas.append([])
            if layer == 0:
                for node in range(self.nNodesPerLayer[-1]):
                    if self.activationFunctions[-1] == 0:
                        error = -self.expectedOutput[node] + self.outputs[-1][node]
                        self.Deltas[layer].append(error*self.tanhPrime(self.inputs[-1][node]))
                        for previousNode in range(self.nNodesPerLayer[-2]):
                            adjMatrices[-1][node][previousNode] = self.adjMatrices[-1][node][previousNode] \
                                                                  - self.alpha * self.Deltas[layer][node] \
                                                                  * self.outputs[-2][previousNode]

                    elif self.activationFunctions[-1] == 1:
                        error = -self.expectedOutput[node] + self.outputs[-1][node]
                        self.Deltas[layer].append(error*self.sigmoidPrime(self.inputs[-1][node]))
                        for previousNode in range(self.nNodesPerLayer[-2]):
                            adjMatrices[-1][node][previousNode] = self.adjMatrices[-1][node][previousNode] \
                                                                  - self.alpha * self.Deltas[layer][node] \
                                                                  * self.outputs[-2][previousNode]

                    elif self.activationFunctions[-1] == 2:
                        error = -self.expectedOutput[node] + self.outputs[-1][node]
                        self.Deltas[layer].append(error*self.ReLuBoundPrime(self.inputs[-1][node]))
                        for previousNode in range(self.nNodesPerLayer[-2]):
                            adjMatrices[-1][node][previousNode] = self.adjMatrices[-1][node][previousNode] \
                                                                  - self.alpha * self.Deltas[layer][node] \
                                                                  * self.outputs[-2][previousNode]
                    elif self.activationFunctions[-1] == 3:
                        D = (-expectedOutput[node]*(1-self.outputs[-1][node]) + \
                                                      (1-self.expectedOutput[node])*self.outputs[-1][node])
                        self.Deltas[layer].append(D)
                        for previousNode in range(self.nNodesPerLayer[-2]):
                            adjMatrices[-1][node][previousNode] = self.adjMatrices[-1][node][previousNode] \
                                                      - self.alpha*( -expectedOutput[node]*(1-self.outputs[-1][node]) \
                                                      * self.outputs[-2][previousNode] + \
                                                      (1-self.expectedOutput[node])*self.outputs[-1][node]\
                                                      * self.outputs[-2][previousNode])

            else:
                errors = np.dot(np.transpose(self.adjMatrices[-layer]), self.Deltas[layer-1])
                for node in range(self.nNodesPerLayer[-1-layer]):
                    # errorList = []
                    # for k in range(self.nNodesPerLayer[-layer]):
                    #     errorList.append(self.adjMatrices[-layer][k][node] * self.Deltas[layer-1][k])
                    # error = sum(errorList)
                    if self.activationFunctions[-2] == 0:
                        self.Deltas[layer].append(errors[node]*self.tanhPrime(self.inputs[-1-layer][node]))
                    if self.activationFunctions[-2] == 1:
                        self.Deltas[layer].append(errors[node]*self.sigmoidPrime(self.inputs[-1-layer][node]))
                    if self.activationFunctions[-2] == 2:
                        self.Deltas[layer].append(errors[node]*self.ReLuBoundPrime(self.inputs[-1-layer][node]))
                    if layer != self.nLayers - 1:
                        for previousNode in range(self.nNodesPerLayer[-2]):
                            adjMatrices[-1-layer][node][previousNode] = self.adjMatrices[-1-layer][node][previousNode] \
                                                                       - self.alpha*self.Deltas[layer][node] \
                                                                       * self.outputs[-2-layer][previousNode]
                    else:
                        for previousNode in range(self.nInput):
                            adjMatrices[-1 - layer][node][previousNode] = self.adjMatrices[-1 - layer][node][previousNode] \
                                                                               - self.alpha * self.Deltas[layer][node] \
                                                                               * self.input[previousNode]

        self.adjMatrices = adjMatrices






# get RGB values for the training image
im = imageio.imread('training_color3.jpg')
im = im[:, :, :3]


training = []
test = []
val = []
# get the corresponding greyscale values for the training image and separate the data in three sets.
# Training, Testing, Validation 80% is used for training, 10% for testing and 10% for validation at random.
im_greyscale = []
im = im/255
for i in range(len(im)):
    im_greyscale.append([])
    for j in range(len(im[i])):
        # im[i][j] = [im[i][j][k]/255 for k in range(3)]
        tmp = im[i][j]
        im_greyscale[i].append(tmp[0]*0.21 + tmp[1]*0.72 + tmp[2]*0.07)
        r = np.random.rand(1)
        if r <= 1:
            training.append([i, j])
        # elif r <0.9:
        #     test.append([i, j])
        # else:
        #     val.append([i, j])


f = plt.figure()
ax = f.add_subplot(111)
ax.imshow(im_greyscale,cmap='gray')
# train the data
error = 0

nPixels = len(im)*len(im[0])


nn = NeuralNet(2, [15, 3])
nn.activationFunctions = [0,0]
nn.initializeNN(.1)
nn.errorType = 'logistic'
errors = []
Clusters = nn.findClusterCenters(im)
centers = Clusters.centers
labels = Clusters.labels
Clusters = None


fig = plt.figure()
ax1 = fig.add_subplot(111)

imDescrete = []
for i in range(len(im)):
    imDescrete.append([])
    for j in range(len(im[0])):
        imDescrete[i].append(centers[labels[i][j]])

fig = plt.figure()
axImDescr = fig.add_subplot(111)
axImDescr.imshow(imDescrete)
errors = []
for epoch in range(int(nPixels)):
    print(epoch)
    inputPixelRow = np.random.randint(1,len(im)-1)
    inputPixelCol = np.random.randint(1,len(im[0])-1)
    while [inputPixelRow, inputPixelCol] in test or [inputPixelRow, inputPixelCol] in val:
        inputPixelRow = np.random.randint(1, len(im) - 1)
        inputPixelCol = np.random.randint(1, len(im[0]) - 1)

    inputGrey = []
    inputGrey.append(im_greyscale[inputPixelRow-1][inputPixelCol-1])
    inputGrey.append(im_greyscale[inputPixelRow-1][inputPixelCol])
    inputGrey.append(im_greyscale[inputPixelRow-1][inputPixelCol+1])
    inputGrey.append(im_greyscale[inputPixelRow][inputPixelCol-1])
    inputGrey.append(im_greyscale[inputPixelRow][inputPixelCol])
    inputGrey.append(im_greyscale[inputPixelRow][inputPixelCol+1])
    inputGrey.append(im_greyscale[inputPixelRow+1][inputPixelCol-1])
    inputGrey.append(im_greyscale[inputPixelRow+1][inputPixelCol])
    inputGrey.append(im_greyscale[inputPixelRow+1][inputPixelCol+1])

    expectedOutput = np.zeros((nn.nColors,1))
    expectedOutput[labels[inputPixelRow][inputPixelCol]] = 1
    nn.SimulateNN(inputGrey, expectedOutput)
    errors.append(nn.error)
    nn.errorBackpropagation()
    # plt.scatter(epoch, nn.error)

plt.plot(range(len(errors)), errors)

testErrors = []
for i in range(len(test)):
    inputPixelRow = test[i][0]
    inputPixelCol = test[i][1]
    if inputPixelRow != 0 and inputPixelRow != len(im)-1 and inputPixelCol != 0 and inputPixelCol!= len(im[0])-1:
        testinputGrey = []
        testinputGrey.append(im_greyscale[inputPixelRow-1][inputPixelCol-1])
        testinputGrey.append(im_greyscale[inputPixelRow-1][inputPixelCol])
        testinputGrey.append(im_greyscale[inputPixelRow-1][inputPixelCol+1])
        testinputGrey.append(im_greyscale[inputPixelRow][inputPixelCol-1])
        testinputGrey.append(im_greyscale[inputPixelRow][inputPixelCol])
        testinputGrey.append(im_greyscale[inputPixelRow][inputPixelCol+1])
        testinputGrey.append(im_greyscale[inputPixelRow+1][inputPixelCol-1])
        testinputGrey.append(im_greyscale[inputPixelRow+1][inputPixelCol])
        testinputGrey.append(im_greyscale[inputPixelRow+1][inputPixelCol+1])
        expectedOutput = np.zeros((nn.nColors, 1))
        expectedOutput[labels[inputPixelRow][inputPixelCol]] = 1
        nn.SimulateNN(testinputGrey, expectedOutput)
        testErrors.append(nn.error)



fig2 = plt.figure()
ax21 = fig2.add_subplot(111)
plt.plot(range(len(testErrors)), testErrors)

im = None

im = imageio.imread('test_color.png')
im = im[:, :, :3]

im_greyscale = []
im = im/255
for i in range(len(im)):
    im_greyscale.append([])
    for j in range(len(im[i])):
        # im[i][j] = [im[i][j][k]/255 for k in range(3)]
        tmp = im[i][j]
        im_greyscale[i].append(tmp[0]*0.21 + tmp[1]*0.72 + tmp[2]*0.07)

im_reconstructed = []
for i in range(1, len(im_greyscale)-1):
    im_reconstructed.append([])
    for j in range(1, len(im_greyscale[0])-2):
        im_reconstructed[i-1].append([])
        inputPixelRow = i
        inputPixelCol = j
        inputGrey = []
        inputGrey.append(im_greyscale[inputPixelRow - 1][inputPixelCol - 1])
        inputGrey.append(im_greyscale[inputPixelRow - 1][inputPixelCol])
        inputGrey.append(im_greyscale[inputPixelRow - 1][inputPixelCol + 1])
        inputGrey.append(im_greyscale[inputPixelRow][inputPixelCol - 1])
        inputGrey.append(im_greyscale[inputPixelRow][inputPixelCol])
        inputGrey.append(im_greyscale[inputPixelRow][inputPixelCol + 1])
        inputGrey.append(im_greyscale[inputPixelRow + 1][inputPixelCol - 1])
        inputGrey.append(im_greyscale[inputPixelRow + 1][inputPixelCol])
        inputGrey.append(im_greyscale[inputPixelRow + 1][inputPixelCol + 1])

        expectedOutput = np.zeros((3, 1))
        #expectedOutput[labels[inputPixelRow][inputPixelCol]] = 1
        nn.SimulateNN(inputGrey, expectedOutput)
        label = nn.outputs[-1].index(max(nn.outputs[-1]))
        im_reconstructed[i-1][j-1]= centers[label]



fig = plt.figure()
ax1 = fig.add_subplot(111)
ar = np.array(im_reconstructed)
plt.imshow(ar)

