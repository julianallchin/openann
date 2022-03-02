import os

import numpy as np

from context import ann

dir = os.path.dirname(__file__)
training_set = "./mnist_train.csv"  # short training set


"""Create instance of neural network """

learningRate = 0.001
nn = ann.NeuralNetwork("weights.npy", [784, 200, 10], "sigmoid", "quadratic")
# nn.load('weights.txt')

"""Train the ANN"""
# load the mnist training data CSV file into a list
filename = os.path.join(dir, training_set)
training_data_file = open(filename, 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()
epochs = 725
limit = 1000

print()
print("...TRAINING...")
for e in range(epochs):
    i = 0
    debug = '\rEpochs: %i \tAccuracy: %f' % (e, (1-nn.lastvariance)*100)
    print(debug, end="", flush=True)

    for i in range(len(training_data_list)-1):

        if i == limit:
            break
        i += 1

        record = training_data_list[i]

        try:
            # split the record by the ',' commas
            all_values = record.split(',')

            inputs = (np.asfarray(
                all_values[1:]) / 255.0 * 0.99) + 0.01  # scale
            # create the target output values (all 0.01, except ans = 0.99)
            targets = np.zeros(10) + 0.01
            # all_values[0] answer for input
            targets[int(all_values[0])] = 0.99
            X = np.array(inputs, ndmin=2)  # create a 784 x 1 vector from input
            nn.train(X, targets)

        except KeyboardInterrupt:
            print('Interrupted')
            nn.save("weights.npy")
            exit(1)


nn.save("weights.npy")
print()
print("Weight saved")
