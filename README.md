# openann

An Artificial Neural Network written in python only using the linear algebra tool [NumPy](https://numpy.org/). It was created as an experiment of a simple, diverse ANN which was user friendly and concise.

### Usage

```python

import openann.ann

# Initialization
weightsFile = "weights.npy"
nodes = [781, 50, 30,..., 10] # input, hidden1, hidden2..., output
activationFunction="sigmoid"
costFunction="quadratic"
lr = 0.001 # Learning rate

# Create network
#  if weightsFile == "" then training starts over.
#  otherwise, weightsFile is used as the starting point for continued training
#  activationFunction can be "sigmoid" (others will be added in the future)
#  costFunction can be "quadratic" or "crossentropy"
nn = ann.NeuralNetwork(weightsFile, nodes, activationFunction, costFunction)

# Learning rate can be changed whenever desired
nn.setLearningRate(lr)

# Train
for i in training_data.range():
    nn.train(i.inputs, i.correctOutput) # Backpropagation
    print(nn.accuracy)
nn.save('weights.npy') # save the knowledge for later

# Guess - That is, when you want to use a trained network
#         If done at a later time, then be sure to create the ann first.
nn.load('weights.npy') # Loads the previous weights
inputs = test_file.readlines() # Simplified
outputs = nn.guess(inputs) 

```

