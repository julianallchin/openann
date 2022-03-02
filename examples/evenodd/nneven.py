#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 19:38:54 2019
@author: Julian Allchin
"""
import random
import numpy as np
from context import ann

def bin_array(num, m):
    """Convert a positive integer num into an m-bit bit vector"""
    return np.array(list(np.binary_repr(num).zfill(m))).astype(np.int8)


"""Create instance of neural network"""
bitlength = 16
print("This ANN determines whether counting number is even or odd from 0 -", 2**bitlength-1)
inputNodeCount = bitlength
hiddenNodeCount = 40
outputNodeCount = 2
learningRate = 0.01
nn = ann.NeuralNetwork(
    "", [inputNodeCount, hiddenNodeCount, outputNodeCount], "sigmoid", "quadratic")
nn.setLearningRate(learningRate)
trainingCount = 1000

"""Create a list of random number from 0 - 2^bitlength - 1"""
num = []
for i in range(1, trainingCount):
    numx = random.randint(0, 2**bitlength-1)
    num.extend([numx])

""" Train network"""
epochs = 10
print("Training ANN for", epochs, "epochs and", trainingCount, "random numbers")
for e in range(epochs):
    for i in range(trainingCount):
        inputs = np.array(bin_array(num[i-1], bitlength))
        targets = np.zeros(outputNodeCount)
        targets[num[i-1] % 2] = 1
        X = np.array(inputs, ndmin=2)  # create input array for ANN
        nn.train(X, targets)
print("ANN finished training")

"""Test network"""
print("Testing ANN with new random numbers")
scorecard = []
for i in range(1, 25):
    num = random.randint(0, 2**bitlength-1)
    bin_num = bin_array(num, bitlength)
    inputs = np.array(bin_num)
    # see what the network says is the correct answer
    outputs = nn.guess(inputs)
    # the index of the highest value corresponds to the answer
    indextoans = np.argmax(outputs)
    if (indextoans == 1 and num % 2 == 1):
        # network's answer matches correct answer, add 1 to scorecard
        print(num, "odd")
        scorecard.append(1)
    elif indextoans == 0 and num % 2 == 0:
        # network's answer doesn't match correct answer, add 0 to scorecard
        print(num, "even")
        scorecard.append(1)
    else:
        print(num, "incorrect")
        scorecard.append(0)
        pass
    pass

"""Calculate the performance score, the fraction of correct answers"""
print("correct answers:", scorecard)
scorecard_array = np.asarray(scorecard)
print("performance =", scorecard_array.sum()
      * 100. / scorecard_array.size, "%")
