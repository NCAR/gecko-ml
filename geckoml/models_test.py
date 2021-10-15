from geckoml.models import DenseNeuralNetwork
import numpy as np
import pandas as pd


np.random.seed(9)

#test run, test shape, test prediction replication
def test_denseneuralnetwork():
    nn = DenseNeuralNetwork()
    x = np.random.random(size=(100, 3))
    nn.build_neural_network(3,5)
    y = nn.predict(x)
    z = nn.predict(x)
    assert np.array_equal(y,z)
    assert not np.isnan(y).any()
    assert y.shape[-1] == 5

#test load data  
def test_loaddata():
    df = pd.read_csv("test_data/test_data.csv", skiprows = 3)
    assert not df.isnull().values.any()
    assert not df.isna().values.any()
    assert df.shape[0] == 1437
    assert df.shape[1] == 128
   
    
if __name__ == "__main__":
    test_denseneuralnetwork()
