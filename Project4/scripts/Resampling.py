import numpy as np
import pdb
import random

from numpy.core.fromnumeric import size
from numpy.lib.stride_tricks import as_strided
class Resampling:

    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 4.3]
    """

    def __init__(self):
        """
        TODO : Initialize resampling process parameters here
        """

    def low_variance_sampler(self, X_bar):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """

        X_bar_resampled = []
        M = X_bar.size()
        if M == 0: return X_bar
        r = np.random.uniform(0,1.0/M)
        c = X_bar[0][3]
        i = 1

        for m in range(0,M):
            U = r + 1.0*m/M
            while U > c:
                i = i + 1
                c = c + X_bar[i][3]
            X_bar_resampled.append(X_bar[i])
        
        """
        TODO : Add your code here
        """

        X_bar_resampled = np.asarray(X_bar_resampled)
        
        return X_bar_resampled

if __name__ == "__main__":
    pass