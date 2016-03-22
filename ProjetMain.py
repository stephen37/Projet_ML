from __future__ import division
from __future__ import print_function

import numpy as np
import random

# Getting all datas
#------------------
def unpickle(f):
    import cPickle
    fo = open(f, 'rb')
    d = cPickle.load(fo)
    fo.close()
    return d

database = unpickle('cifar-10-batches-py/data_batch_1')
testbase = unpickle('cifar-10-batches-py/test_batch')
database_img = database['data']
database_labels = database['labels']
label_list = unpickle('cifar-10-batches-py/batches.meta')
label_names = label_list['label_names']

# Define Constants
#------------------
ITERATIONS = 100
OUTPUT_PATH = "./classifiers_models"
INPUT_PATH = "./cifar-10-batches-py"
TRAINING_RATIO = 1

# Define class Perceptron
#-------------------------
class Perceptron():

    def __init__(self, nbClasses, training_data,
                 training_labels, iterations, training_ratio, testing_data, testing_labels):

        self.nbClasses = nbClasses
        # We initialize the weight with 0 for our entire data
        self.w = [np.zeros(3073) for _ in range(nbClasses)]
        self.iterations = iterations
        self.training_data = training_data
        self.train_targets = training_labels
        self.testing_data = testing_data
        self.testing_targets = testing_labels

        self.training_ratio = training_ratio
        random.shuffle(list(self.training_data))
        #Here, we create the train set.
        self.train_set = [np.append(d, 1) for d in self.training_data]
        self.train_set_targets = self.train_targets[:int(len(self.train_targets) * self.training_ratio)]

        # Here, we create the tests set.
        self.test_set = [np.append(d, 1) for d in self.testing_data]
        self.test_set_targets = self.testing_targets[:int(len(self.testing_targets) * self.training_ratio)]

        self.error_count = 0

    def learning(self, lr=1.0):
        print("The learning process is starting")
        for i in range(self.iterations):
            self.error_count = 0
            #print((i/self.iterations) * 100, "%")
            for xi, yi in zip(self.train_set, self.train_set_targets):
                y = [np.dot(w, xi) for w in self.w]
                prediction = np.argmax(y)
                #print(prediction)
                if prediction != yi :
                    self.error_count += 1
                    self.w[prediction] -= xi * lr
                    self.w[yi] += xi * lr
        #self.error = self.error_count / self.iterations
            self.testing(i)
            lr = lr*0.95
        print("Error for the last learning iteration", (self.error_count/len(self.train_set)) * 100, '%')
        print("Fin")

    def testing(self, i=0):
        #print("The testing process is starting")
        self.error_count = 0
        for xi, yi in zip(self.test_set, self.test_set_targets):
            y = [np.dot(w, xi) for w in self.w]
            prediction = np.argmax(y)
            if prediction != yi :
                self.error_count += 1
        print("Error for the testing in iteration", i , ' : ', (self.error_count/len(self.test_set)) * 100, '%')
        print("--------------------------")


#, training_data, training_labels, iterations, training_ratio
training_data = database_img
train_targets = database_labels
testing_data = testbase['data']
testing_targets = testbase['labels']

perceptron = Perceptron(10, training_data, train_targets, ITERATIONS, TRAINING_RATIO, testing_data, testing_targets)

perceptron.learning()

perceptron.testing()
