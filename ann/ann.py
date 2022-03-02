#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 19:38:54 2019
@author: Julian Allchin
"""

import numpy as np

"""ACTIVATION Functions"""
# Different activation function and its derivative
def sigmoid(x):
    return (1/(1+np.exp(-x)))
def derivativeSigmoid(z):
     sig=sigmoid(z)
     return (sig * (1 - sig))
 
""" COST Functions"""
# quadratic cost function and its derivative
def quadraticCost(y, hypothesisY):
    return 0.5 * np.square(y - hypothesisY)    
def quadraticCostPrime(y, hypothesisY):
    return ( y - hypothesisY)

# Cross Entropy cost function and its derivative    
def crossEntropyCost(y, hypothesisY):
    return - (y * np.log(hypothesisY) + (1 - y)* np.log(1-hypothesisY))
def crossEntropyCostPrime(y, hypothesisY):
    return (hypothesisY - y)/((1 - hypothesisY)*hypothesisY)


"""Artificial Neural Network Class"""
class NeuralNetwork:
     
    # Initialize a neural network system.
    # rehydrate file is either "" or the file name of the saved network
    # if rehydrating, network is reconsituted from file; other parameters ignored
    def __init__(self, rehydrateFile, NetSize, actFunc, costFunc):

        self.error = []
        self.lr = .01  # default learning rate
        self.lastvariance = 100
        self.actFunc = actFunc
        self.costFunc = costFunc
        self.netSize = NetSize
        

        if self.load(rehydrateFile):
            # Create the neural network biases and weights arrays
#           self.biases = [np.ones(i) for i in NetSize[1:]]
            self.biases = [np.zeros(i) for i in NetSize[1:]]
            self.weights=[np.random.randn(j, i)*0.1 for i, j in zip(NetSize[:-1], NetSize [1:])] 
           
        self.numberOfLayers = len(self.netSize)
      
        # Print out the sizes of the layers
        print("Input layer size: "+ str(self.netSize[0]))
        for i in range(len(self.weights)):
            print("W" + str(i),"weight matrix of size " + str(self.weights[i].shape))
        print("Output layer size: "+ str(self.netSize[-1]))
        
        # Set Cost function and its derivative
        if costFunc == "quadratic":
            print("Cost Function: Quadratic")
            self.f_cost = lambda y,a: quadraticCost(y, a)
            self.df_cost = lambda y,a: quadraticCostPrime(y, a)
        elif costFunc == "crossentropy":
            print("Cost Function: Cross Entropy")
            self.f_cost = lambda y,a: crossEntropyCost(y, a)
            self.df_cost = lambda y,a: crossEntropyCostPrime(y, a)
            
        
        # Finally, set the activation function and its derivative
        if actFunc == "sigmoid":
            print("Activation Function: Sigmoid")
            self.f_activation = lambda x: sigmoid(x)
            self.df_activation = lambda x: derivativeSigmoid(x)
     
    def setLearningRate (self, learningRate):
        self.lr = learningRate

    def accuracy(self):
        return (1-self.lastvariance)

    def variance(self, y):
        return np.sum(np.square(y - self.y_hat))

    # Forward propagate an input during training through the network
    def forwardProp(self, X):
        self.layerOutput = [X]
        self.layerInput= [X]
        activation = X.T
        for b, w in zip(self.biases, self.weights):
           z = np.dot(w, activation) + np.array(b, ndmin=2).T
           self.layerInput.append(z.T)
           activation = self.f_activation(z)
           self.layerOutput.append(activation.T)           
        self.y_hat = self.layerOutput[-1] # Set y_hat to the last (output) nodes

    # Backpropagate the error through the network. Use Calculus chain rule
    def backProp(self, X, y):
        self.lastvariance = self.variance(y)
        # For each layer calculate the error
        for i in range(self.numberOfLayers-2, -1, -1):
            if i == self.numberOfLayers-2:
                errorsum = (self.df_cost(y,self.y_hat) * self.df_activation(self.layerInput[-1])).T
            else:
                errorsum = (np.dot(self.weights[i+1].T, errorsum) * self.df_activation(self.layerInput[i+1]).T)

            self.delta = np.dot(errorsum, self.layerOutput[i])

            # Mutliplying by the learning rate
            self.delta = self.delta * self.lr

            # Nudging it
            self.weights[i] += self.delta

    # train the neural network
    def train(self, X, y):
        self.forwardProp(X)
        self.backProp(X, y)

    # after training save weights
    def save(self, fname):
        with open(fname,'wb') as f:
            np.save(f, self.netSize)
            np.save(f, self.actFunc)
            np.save(f, self.costFunc)
            np.save(f, self.biases)
            np.save(f, self.weights)
        f.close()

    # after training load saved weights
    def load(self, file):
        try:
            with open(file, 'rb') as f:
                self.netSize = np.load(f,allow_pickle=True)
                self.actFunc = np.load(f, allow_pickle=True)
                self.costFunc = np.load(f, allow_pickle=True)
                self.biases = np.load(f, allow_pickle=True)
                self.weights = np.load(f, allow_pickle=True)
                f.close()
                return 0
        except IOError:
            return 1

    # Given a trained network, test a new input
    def guess(self, X):
        a = X
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) +b)
        return a