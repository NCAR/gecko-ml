from pyDOE import lhs
import numpy as np
import pandas as pd
from scipy.stats import loguniform
import matplotlib.pyplot as plt

def main():
    seed = 2345
    params = {"temp": (240.0, 315.0),
              "sza": (0, 90),
              "poa": (0.01, 100),
              "o3": (1.0, 150.0),
              "nox": (0.1, 10),
              "oh": (1, 10)
             }
    param_names = list(params.keys())
    print(param_names)
    log_params = ["poa", "nox"]
    experiments = 2000
    out_file = "gecko_experiments_djg_20190306.csv"
    np.random.seed(seed)
    sample_vals = lhs(len(param_names), samples=experiments)
    scaled_vals = np.zeros(sample_vals.shape)
    for p, p_name in enumerate(param_names):
        if p_name in log_params:
            scaled_vals[:, p] = loguniform.ppf(sample_vals[:, p], a=params[p_name][0], b=params[p_name][1])
        else:
            scaled_vals[:, p] = sample_vals[:, p] * (params[p_name][1] - params[p_name][0]) + params[p_name][0]
        print(p_name, params[p_name])
        print(scaled_vals[:, p].min(), scaled_vals[:, p].max())
    scaled_frame = pd.DataFrame(scaled_vals, columns=param_names)
    scaled_frame.to_csv(out_file, index_label="ID")


if __name__ == "__main__":
    main()
