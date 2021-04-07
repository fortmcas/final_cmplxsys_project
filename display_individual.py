import pickle


class SimpleNeuralNet:
    def activation_function(self, x):
        return 1 / (1 + np.exp(-x))

    def deepcopy(self):
        new_net = SimpleNeuralNet(
            self.num_inputs, self.num_outputs, self.layer_node_counts
        )
        new_net.layers = [np.copy(layer) for layer in self.layers]
        return new_net

    def execute(self, input_vector):
        assert len(input_vector) == self.num_inputs, "wrong input vector size"

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
            self.layers.append(
                np.random.uniform(-100, 100, size=(last_num_neurons + 1, nc))
            )
            last_num_neurons = nc


with open("best_so_far.pckl", "rb") as f:
    best = pickle.load(f)
print(best.layers)
