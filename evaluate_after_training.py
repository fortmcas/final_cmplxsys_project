import numpy as np
from neural_net import SimpleNeuralNet
np.seterr(over="ignore", divide="raise")
import subprocess
import random
import pickle

def get_network_output(simple_net, input_set):
    return_list = []
    for test_index in range(len(input_set)):
        test_output = simple_net.execute(input_set[test_index])
        return_list.append([list(test_output)])
    return return_list

def get_network_fitness(simple_net, input_set, target_output_set):
    assert len(input_set) == len(target_output_set)
    total_distance = 0
    for test_index in range(len(input_set)):
        test_output = simple_net.execute(input_set[test_index])
        target_output = target_output_set[test_index]
        distances = np.linalg.norm(test_output - target_output)
        total_distance += np.sum(distances)

    return -total_distance

filename = "most_fit.pckl"
with open(filename, "rb") as f:
    most_fit_individual = pickle.load(f)

"""    
Output Code: 0 = Benign
             1 = Attack
"""
with open("benign4.pckl", "rb") as f:
    benign4 = pickle.load(f)
with open("benign5.pckl", "rb") as f:
    benign5 = pickle.load(f)
with open("dos3.pckl", "rb") as f:
    dos3 = pickle.load(f)
with open("infil2.pckl", "rb") as f:
    infil2 = pickle.load(f)
with open("sqlinjection2.pckl", "rb") as f:
    sqlinjection2 = pickle.load(f)
with open("bot1.pckl", "rb") as f:
    bot1 = pickle.load(f)
with open("bf2.pckl", "rb") as f:
    bf2 = pickle.load(f)

input_set = [benign4, benign5, dos3, infil2, sqlinjection2, bot1, bf2]

desired_output_set = [0, 0, 1, 1, 1, 1, 1]
print(get_network_fitness(most_fit_individual, input_set, desired_output_set))
achieved_output_set = get_network_output(most_fit_individual, input_set)
print("Desired Output Set: {}".format(desired_output_set))
print("Achieved Output Set: {}".format(achieved_output_set))
