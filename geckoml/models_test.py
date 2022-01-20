from geckoml.models import DenseNeuralNetwork, GRUNet
import numpy as np
import pandas as pd
import torch

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


def test_gruneuralnetwork():
    nn = GRUNet(1215, 1, 0.0)
    nn.build(9, 3)
    nn.eval()
    x = torch.from_numpy(np.random.random(size=(2, 9)).astype(np.float32))
    y = torch.from_numpy(np.random.random(size=(2, 3)).astype(np.float32))
    with torch.no_grad():
        h = nn.init_hidden(x)
        p, h1 = nn(x, h)
        p2, h2 = nn(x, h)
    assert np.array_equal(p.numpy(), p2.numpy())
    assert np.array_equal(h1.numpy(), h2.numpy())
    assert not np.isnan(y).any()
    assert y.shape[-1] == 3


# test load data
def test_loaddata():
    df = pd.read_csv("./test_data/test_data.csv", skiprows=3)
    assert not df.isnull().values.any()
    assert not df.isna().values.any()
    assert df.shape[0] == 1437
    assert df.shape[1] == 128
