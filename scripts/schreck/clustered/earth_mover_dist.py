from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from numba import jit

@jit
def EarthMoverDist2D(Y1, Y2): 
    # https://stackoverflow.com/questions/57562613/python-earth-mover-distance-of-2d-arrays
    d = cdist(Y1, Y2)
    assignment = linear_sum_assignment(d)
    return d[assignment].sum() / Y1.shape[0]