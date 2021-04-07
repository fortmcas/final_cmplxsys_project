import pickle
from neural_net import SimpleNeuralNet

"""
Prints layers of most fit neural network weights
"""
if __name__ == "__main__":
    with open("most_fit.pckl", "rb") as f:
        best = pickle.load(f)
    print(best.layers)
