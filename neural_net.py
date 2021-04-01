import numpy as np
np.seterr(over='ignore', divide='raise')
from matplotlib import pyplot as plt
import subprocess
import random

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

def get_network_fitness(simple_net, input_set, target_output_set):
    assert(len(input_set) == len(target_output_set))
    total_distance = 0

    for test_index in range(len(input_set)):
        test_output = simple_net.execute(input_set[test_index])
        target_output = output_set[test_index]
        distances = np.linalg.norm(test_output - target_output)
        total_distance += np.sum(distances)

    return -total_distance

def tournament_selection(population, input_set, target_set, fit_func, tournament_size=3):
    sample_pop = np.random.choice(population, size=tournament_size)
    sample_pop_fitness = [fit_func(p, input_set, target_set) for p in sample_pop]
    winner_idx = np.argmax(sample_pop_fitness)
    
    return sample_pop[winner_idx]


def mutate_network(simple_net, mutation_rate=0.001):
    if np.random.random() <= mutation_rate:
        for layer_to_mut in simple_net.layers:
            dims = layer_to_mut.shape
            selected_location = []
            for dim in dims:
                selected_location.append(random.choice(range(dim)))
            layer_to_mut[selected_location[0], selected_location[1]] = np.random.uniform(-100,100)

"""    
This code will work even without a mutation function, but you'll not be introducing any variation into the population. Once you have a working mutation function, you'll have more fun. 

For this problem, we're going to try to evolve a solution to the last question from the last worksheet. 

| Input 1 | Input 2 | Output |
|---------|---------|--------|
| 0       | 0       | 0      |
| 0       | 1       | 1      |
| 1       | 0       | 1      |
| 1       | 1       | 0      |
"""
input_set = [[0,0],
             [0,1],
             [1,0],
             [1,1]]

output_set = [[0], 
              [1],
              [1], 
              [0]]


pop_size = 100
num_generations = 100
mutation_rate = 0.05

population = [ SimpleNeuralNet(num_inputs=2, 
                               num_outputs=1, 
                               layer_node_counts=[5])
              for i in range(pop_size)]

avg_fitnesses = []
gen = 0
this_gen_avg_fitness = -1000
while (this_gen_avg_fitness < -0.001):
    print(gen)
    selected_individuals = [tournament_selection(population, 
                                                 input_set, 
                                                 output_set, 
                                                 get_network_fitness).deepcopy()
                            for _ in range(pop_size)]
    for individual in selected_individuals:
      mutate_network(individual, mutation_rate)
    
    this_gen_avg_fitness = np.mean([get_network_fitness(idv, input_set, output_set) for idv in selected_individuals])
    print(this_gen_avg_fitness) 
    avg_fitnesses.append(this_gen_avg_fitness)
    
    population = selected_individuals
    gen+=1

highest_fitness = -1000
most_fit_individual = None
for idv in population:
    this_fitness = get_network_fitness(idv, input_set, output_set)
    if this_fitness > highest_fitness:
        highest_fitness = this_fitness
        most_fit_individual = idv

print(most_fit_individual.layers)
print(highest_fitness)

plt.plot(avg_fitnesses)
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.savefig('plot.png')
subprocess.run(["./move_plot.sh"])
