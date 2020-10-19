from geckoml.models import DenseNeuralNetwork
import numpy as np

import warnings
warnings.filterwarnings("ignore")

def test_denseneuralnetwork():
    nn = DenseNeuralNetwork()
    x = np.random.random(size=(100, 3))
    nn.build_neural_network(3,5)
    y = nn.predict(x)
    assert y.all()
   
    
def test_nnshape():
    nn = DenseNeuralNetwork()
    x = np.random.random(size=(100, 3))
    nn.build_neural_network(3,5)
    y = nn.predict(x)
    assert y.shape[-1] == 5
    
    
    
if __name__ == "__main__":
    test_denseneuralnetwork()
