import numpy as np
np.seterr(over='ignore', divide='raise')
from matplotlib import pyplot as plt
import subprocess
import random
import pickle

class SimpleNeuralNet():
    def activation_function(self, x):
        return 1/(1+np.exp(-x))
    
    def deepcopy(self):
        new_net = SimpleNeuralNet(self.num_inputs, self.num_outputs, self.layer_node_counts)
        new_net.layers = [np.copy(layer) for layer in self.layers]
        return new_net
    
    def execute(self, input_vector):
        assert len(input_vector) == self.num_inputs ,\
        "wrong input vector size"

        next_v = input_vector

        for layer in self.layers:
            next_v = np.append(next_v, 1)
            next_v = self.activation_function(np.dot(next_v, layer))
        return next_v
        
    def __init__(self, num_inputs, num_outputs, layer_node_counts=[]):
        self.num_inputs = num_inputs
        self.layer_node_counts = layer_node_counts
        self.num_outputs = num_outputs
        self.layers = []
        
        last_num_neurons = self.num_inputs
        for nc in layer_node_counts + [num_outputs]:
            self.layers.append(np.random.uniform(-100, 100, size=(last_num_neurons+1, nc)))
            last_num_neurons = nc

def get_network_output(simple_net, input_set):
    return_list = []
    for test_index in range(len(input_set)):
        test_output = simple_net.execute(input_set[test_index])
        return_list.append([list(test_output)])
    return return_list

def get_network_fitness(simple_net, input_set, target_output_set):
    assert(len(input_set) == len(target_output_set))
    total_distance = 0
    for test_index in range(len(input_set)):
        test_output = simple_net.execute(input_set[test_index])
        target_output = target_output_set[test_index]
        distances = np.linalg.norm(test_output - target_output)
        total_distance += np.sum(distances)

    return -total_distance

filename = 'most_fit.pckl'
with open(filename, 'rb') as f:
    most_fit_individual = pickle.load(f)

"""    
Output Code: [0, 0, 0, 1] = Benign
             [0, 0, 1, 0] = DoS Attack
             [0, 1, 0, 0] = Infiltration Attack
             [1, 0, 0, 0] = SQL Injection Attack
"""
with open('benign4.pckl', 'rb') as f:
    benign4 = pickle.load(f)
with open('benign5.pckl', 'rb') as f:
    benign5 = pickle.load(f)
with open('dos3.pckl', 'rb') as f:
    dos3 = pickle.load(f)
with open('infil2.pckl', 'rb') as f:
    infil2 = pickle.load(f)
with open('sqlinjection2.pckl', 'rb') as f:
    sqlinjection2 = pickle.load(f)

input_set = [benign4,
             benign5,
             dos3,
             infil2,
             sqlinjection2]

desired_output_set = [[0, 0, 0, 1],
                      [0, 0, 0, 1],
                      [0, 0, 1, 0],
                      [0, 1, 0, 0],
                      [1, 0, 0, 0],
                    ]
print(get_network_fitness(most_fit_individual, input_set, desired_output_set))
"""
achieved_output_set = get_network_output(most_fit_individual,input_set)
print("Desired Output Set: {}".format(desired_output_set))
print("Achieved Output Set: {}".format(achieved_output_set))
"""
