from geckoml.models import DenseNeuralNetwork
import numpy as np
import pandas as pd


np.random.seed(9)

def test_denseneuralnetwork():
    nn = DenseNeuralNetwork()
    x = np.random.random(size=(100, 3))
    nn.build_neural_network(3,5)
    y = nn.predict(x)
    assert not np.isnan(y).any()   
 

def test_nnshape():
    nn = DenseNeuralNetwork()
    x = np.random.random(size=(100, 3))
    nn.build_neural_network(3,5)
    y = nn.predict(x)
    assert y.shape[-1] == 5
   

def test_loaddata():
    df = pd.read_csv("/glade/p/cisl/aiml/gecko/ML2019_dodecane_postproc/ML2019_dodecane_ML2019_Exp54.csv", skiprows = 3) 
    assert not df.isnull().values.any()
    assert not df.isna().values.any()
    assert df.shape[0] == 1437
    assert df.shape[1] == 128
 

def test_replicateprediction():
    nn = DenseNeuralNetwork()
    x = np.random.random(size=(100, 3))
    nn.build_neural_network(3,5)
    y = nn.predict(x)
    z = nn.predict(x)
    assert np.array_equal(y,z)
    
    
if __name__ == "__main__":
    test_denseneuralnetwork()
