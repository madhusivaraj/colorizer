# This script:
# Initializes a neural network (nn) with specified parameters and random weights, or load pre-saved weights.
# Loads a training images that must be in the same directory as the script.
# Trains the nn to map windows of pixels of the grayscale version of the training image to the correct RGB values of the window's center pixel.
# Saves the training weights each epoch.
# Records the loss values of training and testing images each epoch.
# Colorizes a grayscale image using the trained nn and prints the colorized image.

import numpy as np
import matplotlib.pyplot as plt
import imageio
import time
import json

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
    if label == 0: # 0 = no act fcn (output = input)
        return z
    elif label == 1: # 1 = sigmoid
        return 1/(1+np.exp(-z))
    elif label == 2: # 2 = ReLU
        return max(0,z)
    elif label == 3: # 3 = leaky ReLU
        if z > 0:
            return z
        else:
            return z*0.001
    elif label == 4: # 4 = tanh
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
    # derivative of activation fcn of a given node
    # label indicates which fcn to use
    if label == 0: # 0 = no act fcn
        return 1
    if label == 1: # 1 = sigmoid
        return z*(1-z)
    if label == 2:
        if z > 0: # 2 = ReLU
            return z
        else:
            return 0
    if label == 3: # 3 = leaky ReLU
        if z > 0:
            return z
        else:
            return z*0.001
    if label == 4: # 4 = tanh
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

def ioMaker(img, L, W, window):
    # takes in color training image and current pixel, returns
    # grayscale nn input vector and expected nn output scaled RGB
    f = img[L][W]/255 # scales output rbg values to between 0 and 1
    x = np.zeros([window, window])
    shift = int((np.floor(window/2)))
    for i in range(window): 
        for j in range(window):
            if L+i-shift < 0 or L+i-shift > len(img)-1: # row out of bounds
                x[i][j] = 0 # nn input value 0 (buffer)
            elif W+j-shift < 0 or W+j-shift > len(img[L])-1: # column out of bounds
                x[i][j] = 0 # nn input value 0 (buffer)
            else: # nn input value is grayscaled pixel
                x[i][j] = img[L+i-shift][W+j-shift][0]*0.21 + img[L+i-shift][W+j-shift][1]*0.72 + img[L+i-shift][W+j-shift][2]*0.07
    xFlat = np.zeros(window**2) # flatten input window to a row
    item = 0
    for i in range(len(x)):
        for j in range(len(x)):
            xFlat[item] = x[i][j]
            item += 1
    xFlat = xFlat/255 # scales inputs grayscale pixels to between 0 and 1
    return xFlat, f

# make color image grayscale
def colorToGray(img):
    imgBW = np.zeros([len(img),len(img[0])])
    for i in range(len(img)): # each row
        if np.remainder(i, 20) == 0:
            print('Grayscaling pixel row',i)
        for j in range(len(img[0])): # each column
            imgBW[i][j] = img[i][j][0]*0.21 + img[i][j][1]*0.72 + img[i][j][2]*0.07 # grayscaled pixel
    return imgBW

# colorize gray image
def colorize(imgBW, window, nodes, weights, biasL, actL, sizePic):
    imgCL = imageio.imread(sizePic) # any image with same pixel dimensions as imgBW
    for i in range(len(imgCL)):
        for j in range(len(imgCL[i])):
            for k in range(len(imgCL[i][j])):
                imgCL[i][j][k] = 0 # replace values of this image with 0 to fill with colorized pixels
    x = np.zeros([window, window])
    shift = int((np.floor(window/2)))
    for L in range(len(imgBW)):
        if np.remainder(L, 20) == 0:
            print('Colorizing pixel row', L)
        for W in range(len(imgBW[L])): # L,W is pixel to be colorized
            for i in range(window): # window around pixel to be colorized
                for j in range(window):
                    LL = int(L+i-shift)
                    WW = int(W+j-shift)
                    if LL < 0 or LL > len(imgBW)-1:
                        x[i][j] = 0 # outside bounds, buffer
                    elif W+j-shift < 0 or W+j-shift > len(imgBW[L])-1:
                        x[i][j] = 0 # outside bounds, buffer
                    else: # BW pixel for input window
                        x[i][j] = imgBW[LL][WW]
            xFlat = np.zeros(window**2)
            item = 0
            for i in range(len(x)): # flatten input window to row
                for j in range(len(x)):
                    xFlat[item] = x[i][j]
                    item += 1
            xFlat = xFlat/255 # scale btwn 0 and 1
            nodes = forwardProp(xFlat, nodes, weights, biasL, actL) # use nn to colorize pixel
            RGB = nodes[len(nodes)-1] # store nn output as RGB
            for i in range(len(RGB)):
                if RGB[i] > 1: # if output too high, make max
                    RGB[i] = 1
                RGB[i] *= 255 # scale RGB
            imgCL[L][W] = RGB # place RGB pixel in image
    return imgCL

def trainingLoss(img, imgCL):
    # returns loss for an entire colorized image
    sumLoss = 0
    N = 1/(len(img)*len(img[0]))
    normSqrt = np.sqrt(N)
    for i in range(len(img)):
        for j in range(len(img[i])):
            for k in range(len(img[i][j])):
                #sumLoss += ((img[i][j][k]-imgCL[i][j][k])**2)/(len(img)*len(img[i]))
                a = img[i][j][k]*normSqrt
                b = imgCL[i][j][k]*normSqrt
                sumLoss += (a**2)-(2*N*img[i][j][k]*imgCL[i][j][k])+(b**2)
    return sumLoss
    


##########################################################################
##########################################################################

# Driver Script

# Start timer
t0 = time.time()

# Initialize network
numL = 3 # number of layers
inLen = 3*3 # length of input vector (should be window squared)
outLen = 3 # length of output vector
biasL = [1, 1, 0] # bias for each layer; 1 = layer has bias, 0 = no bias; last always 0
npL = [(inLen+biasL[0]), 25, outLen] # nodes per layer (includes biases)
actL = [0, 4, 1] # label indicating activation fcn type leading into that layer; 1st is meaningless (see "sigma" fcn)
[nodes, deltas, weights] = nnInitialize(numL, npL) # creates node, delta, and weight lists
lossL = 1 # label indicating loss fcn type (see "loss" and "lossPrime" fcns)
#weights = json.load(open('out.weights_1')) # to load saved weights

# Train and test nn
epochs = 4 # number of times it trains on the whole training set
window = 3 # window size
trnLossList = np.zeros(epochs)
tstLossList = np.zeros(epochs)
valLossList = np.zeros(1)
trainingList = ['flower_1.jpg','flower_2.jpg','flower_3.jpg','flower_4.jpg','flower_5.jpg','flower_6.jpg','flower_7.jpg','flower_8.jpg'] # list of names of training images
testingList = ['flower_9.jpg'] # list of names of testing images
validationList = ['flower_10.jpg']
trainingIndex = np.arange(len(trainingList))
sizePic = validationList[0]# any image with same pixel dimensions as imgBW
for i in range(epochs):
    print('Epoch', i)
    
    # train on training set
    np.random.shuffle(trainingIndex) # randomize order in which training images are pulled
    for j in range(len(trainingList)): # for one image in training set
        print('Training on image', j)
        img = imageio.imread(trainingList[trainingIndex[j]]) # load image
        for L in range(len(img)): # L is pixel row coord
        #for L in range(1):
            if np.remainder(L, 20) == 0:
                print('Training on pixel row:', L, 'Runtime:', time.time()-t0, 'sec')
            for W in range(len(img[L])): # W is pixel column coord
            #for W in range(1):
                [x, f] = ioMaker(img, L, W, window) # x is bw window. f is correct color pixel
                nodes = forwardProp(x, nodes, weights, biasL, actL) # [do fwdprop (return nodes)]
                weights = backProp(nodes, deltas, weights, f) # [do backprop (return weights)]
                
    # save epoch weights
    outpath = "{}_{}".format("out.weights", i)
    with open(outpath, "w") as f:
        f.write(json.dumps(weights))
        print('Epoch', i, 'weights saved')
        
    # calculate average loss across all training images
    trnLoss = 0 # loss on training images with weights after training epoch
    for j in range(len(trainingList)):
        print('Calculating loss on training image', j)
        img = imageio.imread(trainingList[j]) # load image
        imgBW = colorToGray(img)
        imgCL = colorize(imgBW, window, nodes, weights, biasL, actL, sizePic)
        jLoss = trainingLoss(img, imgCL) # loss of colorized image j vs its real image
        trnLoss += jLoss # add to overall training loss
    trnLoss /= len(trainingList) # divide by number of training images
    trnLossList[i] = trnLoss
    
    # calculate average loss across all testing images
    tstLoss = 0
    for j in range(len(testingList)):
        print('Calculating loss on testing image', j)
        img = imageio.imread(testingList[j]) # load image
        imgBW = colorToGray(img)
        imgCL = colorize(imgBW, window, nodes, weights, biasL, actL, sizePic)
        jLoss = trainingLoss(img, imgCL) # loss of colorized image j vs its real image
        tstLoss += jLoss # add to overall training loss
    tstLoss /= len(testingList) # divide by number of training images
    tstLossList[i] = tstLoss
    
    # print last testing image each epoch, to see progression
    fig = plt.figure() # print
    fig.suptitle('Colorized testing image for current epoch')
    ax1 = fig.add_subplot(111)
    ar = np.array(imgCL)
    plt.imshow(ar)
    #plt.show()
    imgSaveName = "{}_{}".format("test_img_epoch", i)
    plt.savefig(imgSaveName)

# save losses
np.save('trnLossList', trnLossList)
np.save('tstLossList', tstLossList)

# Calculate validation error and print final colorized validation image
valLoss = 0
for i in range(len(validationList)):
    print('Calculating loss on validation image', j)
    img = imageio.imread(validationList[j]) # load image
    imgBW = colorToGray(img)
    imgCL = colorize(imgBW, window, nodes, weights, biasL, actL, sizePic)
    jLoss = trainingLoss(img, imgCL) # loss of colorized image j vs its real image
    valLoss += jLoss # add to overall training loss
valLoss /= len(validationList) # divide by number of training images
# save validation losses
np.save('valLossList', valLoss)
# print final colorized validation image
fig = plt.figure() # print
fig.suptitle('Colorized validation image')
ax1 = fig.add_subplot(111)
ar = np.array(imgCL)
plt.imshow(ar)
#plt.show()
imgSaveName = "validation_img"
plt.savefig(imgSaveName)

# Plot losses vs epoch
fig = plt.figure()
fig.suptitle('Losses vs epoch (blue=training, orange=testing)')
plt.plot(trnLossList)
plt.plot(tstLossList)
#plt.show()
imgSaveName = "loss_plot"
plt.savefig(imgSaveName)

#
print(' ')
print('Training loss per epoch', trnLossList)
print('Testing loss per epoch', tstLossList)
print('Validation loss', valLoss)
print('Total runtime', (time.time()-t0)/60, 'min')
print('Done')
