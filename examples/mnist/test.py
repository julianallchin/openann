from context import ann
from matplotlib import pyplot as plt
import numpy as np
import os
dir = os.path.dirname(__file__)


"""Create instance of neural network """
learningRate = 0.001
nn = ann.NeuralNetwork("weights.npy", [784, 200, 10], "sigmoid", "quadratic")
nn.setLearningRate(learningRate)

"""Training set data and testing set data"""

TEST_SET = "./mnist_train_100.csv"       # short test set
FILENAME = os.path.join(dir, TEST_SET)

"""Test ANN on new samples"""
test_data_file = open(
    FILENAME, 'r')  # load the mnist test data CSV file into a list
test_data_list = test_data_file.readlines()
test_data_file.close()
scorecard = []
# go through all the records in the test data set
for record in test_data_list:
    all_values = record.split(',')  # split the record by the ',' commas
    correct_label = int(all_values[0])
    # scale and shift the inputs
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    outputs = nn.guess(inputs)  # ANN's guess as to the number
    # the index of the highest value corresponds to the answer
    label = np.argmax(outputs)
    if label == correct_label:
        # network's answer matches correct answer, add 1 to scorecard
        print(correct_label, "correct")
        scorecard.append(1)
    else:

        # network's answer doesn't match correct answer, add 0 to scorecard
        print(correct_label, label, "wrong")
        plt.imshow(inputs.reshape((28, 28)), cmap='gray')
        plt.show()
        scorecard.append(0)
        pass
    pass

"""Calculate the performance score, the fraction of correct answers"""
print(scorecard)
scorecard_array = np.asarray(scorecard)
print("performance =", scorecard_array.sum()
      * 100. / scorecard_array.size, "%")
