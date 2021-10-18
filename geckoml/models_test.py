from geckoml.models import DenseNeuralNetwork
import numpy as np
import pandas as pd

np.random.seed(9)

def test_denseneuralnetwork():
    nn = DenseNeuralNetwork()
    x = np.random.random(size=(100, 9))
    y = np.random.random(size=(100, 3))
    nn.fit(x, y)
    p = nn.predict(x)
    p2 = nn.predict(x)
    assert np.array_equal(p, p2)
    assert not np.isnan(y).any()
    assert y.shape[-1] == 3

#test load data  
def test_loaddata():
    df = pd.read_csv("./geckoml/test_data.csv", skiprows=3)
    assert not df.isnull().values.any()
    assert not df.isna().values.any()
    assert df.shape[0] == 1437
    assert df.shape[1] == 128
   
    
if __name__ == "__main__":
    test_denseneuralnetwork()
