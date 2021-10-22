from data.generate import generate_dataset
from model.model import Network

if __name__ == "__main__":
    dataset = generate_dataset(100)
    net = Network()
    for (x, y) in dataset:
        print(net.forward(x))



