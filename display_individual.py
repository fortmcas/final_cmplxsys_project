from neural_net import SimpleNeuralNet
import pickle

if __name__ == "__main__":
    with open("most_fit.pckl", "rb") as f:
        best = pickle.load(f)
    print(best.layers)
