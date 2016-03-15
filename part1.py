# coding=utf-8
"""

Learning Multiple Layers of Features from
Tiny Images", Alex Krizhevsky, 2009.
http://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf

"""

# Author : BATIFOL Stephen, M1 Informatique


from __future__ import division
from __future__ import print_function
from pprint import pprint

import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import random



def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

"""
LABELS = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
"""

"""
class CIFAR10 :

    def horizontal_histogram(self, image) :
        return np.histogram(image, bins = range(256))
            

    def mean_image(self, image_list) :
        #mean_array = np.zeros(1024 * 3, dtype=np.uint8)
        mean_array = [0] * 1024 * 3
        #test = True
        for image in image_list :
            '''
            if test : 
                hist, bin_edges = self.horizontal_histogram(image)
                plt.bar(bin_edges[:-1], hist, width = 1)
                plt.xlim(min(bin_edges), max(bin_edges))
                plt.show()
                test = False

            ''' 
            for i in range(len(image)) :
                mean_array[i] += image[i]
        length = len(image_list)
        print("shape image_list", len(image_list) , "shape mean_array", len(mean_array))
        for i in range(len(image_list)) :
            mean_array[i] = mean_array[i] / length
        
        return mean_array

    def show_image_from_list(self, list) :
        img_np = np.array(list).reshape(32, 32, 3)22
        plt.imshow(img_np)
        plt.show()

    def mean_all_images(self, value) :
        image_list = []
        for i in range(len(self.train_labels)) :
            if self.train_labels[i] == value :
                image_list.append(self.train_features[i])
                
        mean = self.mean_image(image_list)
        #self.show_image_from_list(mean)
        return mean

"""
        
#cifar = CIFAR10()
#cifar.load_datasets()
#print(cifar.mean_all_images(4))

#Constants
ITERATIONS = 100
OUTPUT_PATH = "./classifiers_models"
INPUT_PATH = "./cifar-10-batches-py"
TRAINING_RATIO = 0.5

class Perceptron() :

    def __init__(self, threshold, nbClasses) :
        self.threshold = threshold
        self.nbClasses = nbClasses
        self.w = np.zeros(3072) # We initialize the weight with 0 for our entire data
        
    """
    The function load_datasets returns a tuple of
    0 -> train_features
    1 -> train_labels

    On va représenter les pixels par des vecteurs, chaque pixel est représenté par un vecteur de 3072.
    """

    def load_datasets(self) :
        self.train_batches = []
        
        # Load the datasets.
        for batch in range(1,6) :
            self.train_batches.append(unpickle(INPUT_PATH + "/data_batch_%d" % batch))


        self.train_features = np.concatenate(
            [batch["data"].reshape(batch["data"].shape[0], 3, 32, 32)
             for batch in self.train_batches])


        self.train_labels = np.concatenate(
            [np.array(batch["labels"], dtype=np.uint8)
             for batch in self.train_batches])
        self.train_labels = np.expand_dims(self.train_labels, 1)

        data = [elt for elt in zip(self.train_features,self.train_labels) if elt[1] < self.nbClasses]

        return data



    def learning(self, training_data, training_labels, iterations, training_ratio) :
        self.iterations = iterations
        self.training_data = training_data
        self.train_targets = training_labels
        #print("self.train_targets", self.train_targets)
        self.training_ratio = training_ratio
        
        random.shuffle(list(self.training_data))
        
        #Here, we create the train set.
        self.train_set = self.training_data[:int(len(self.training_data) * self.training_ratio)]
        self.train_set_targets = self.train_targets[:int(len(self.train_targets) * self.training_ratio)]
        #pprint(self.train_set_targets)
        # Here, we create the tests set.
        self.test_set = self.training_data[int(len(self.training_data) * self.training_ratio)]
        self.test_set_targets = self.train_targets[int(len(self.train_targets) * self.training_ratio)]

        print("We start the learning process")
        self.error_count = 0
        for i in range(self.iterations) :
            if len(self.train_set) == len(self.train_set_targets) :
                for xi, yi in zip(self.train_set, self.train_set_targets) :
                    y = map(lambda w : np.dot(w, xi), self.w)
                    print(y)
                    prediction = np.argmax(y)
                    #print("prediction", prediction, "yi ", yi)
                    #print("yi[i][1] = ", yi[i][1])
                    """
                    if prediction != yi[i] :
                        self.error_count += 1
                        print(self.w)
                        self.w[prediction] -= xi
                        self.w[yi] += xi
                    """
                    print("prediction", prediction)
            else : 
                print("len(self.train_set_targets) != len(self.train_set_targets)")
        self.error = self.error_count / self.iterations
        print("Error for the learning part", self.error)

perceptron = Perceptron(0.5, 3)
data = perceptron.load_datasets()


training_data = [data[i][0] for i in range(len(data))]
train_targets = [data[i][1] for i in range(len(data))]

perceptron.learning(training_data, train_targets, ITERATIONS, TRAINING_RATIO)