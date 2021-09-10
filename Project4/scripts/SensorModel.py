
import numpy as np
import math
import time
from matplotlib import pyplot as plt
from scipy.stats import norm
from scipy.stats import expon
import pdb

from MapReader import MapReader

class SensorModel:

    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 6.3]
    """

    def __init__(self, occupancy_map):
        """
        Initialize Sensor Model parameters here
        """
        self.map = occupancy_map
        self.Z_MAX = 8183
        self.P_HIT_SIGMA = 250
        self.P_SHORT_LAMBDA = 0.01
        self.Z_PHIT = 1000
        self.Z_PSHORT = 0.01
        self.Z_PMAX = 0.03
        self.Z_PRAND = 100000
        self.K = 180
        self.eps = 1e-6
 
    def beam_range_finder_model(self, z_t1_arr, x_t1):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[out] q : likelihood of a range scan zt1 at time t
        """

        """
        TODO : Add your code here
        """

        # get the z_t^{k_star} by the provided ray-casting algorithm:
        # here is an example:
        x = x_t1[0] + 25 * np.cos(x_t1[2])
        y = x_t1[1] + 25 * np.sin(x_t1[2])
        z_t_k_star = self.rayCast( -90, x_t1[2], x, y )

        # where x, y are the coordinations that need to be calculated by yourself 
        # according to the current location and \theta
        # (note there is a 25cm offset from agent to the laser sensor)

        q = 0
        for k in range(self.K):

            # calculate p_hit

            # calculate p_short

            if z_t1_arr[k] >= 0 and z_t1_arr[k] <= z_t_k_star[k]:
                p_short = self.P_SHORT_LAMBDA * np.exp(-self.P_SHORT_LAMBDA*z_t1_arr[k])
                p_short = p_short / (1 - np.exp(-self.P_SHORT_LAMBDA*z_t_k_star[k]))
            else:
                p_short = 0

            # calculate p_max

            if z_t1_arr[k] >= self.Z_MAX - self.eps:
                p_max = 1.0
            else:
                p_max = 0

            # calculate p_rand

            if z_t1_arr[k] >= 0 and z_t1_arr[k] < self.Z_MAX - self.eps:
                p_rand = 1.0/self.Z_MAX
            else:
                p_rand = 0

            # calculate p

            p = self.Z_PHIT * p_hit + self.Z_PSHORT * p_short + self.Z_PMAX * p_max + self.Z_PRAND * p_rand
            q = q*p
        
        return q


    def rayCast(self, deg, ang, coord_x, coord_y):
        final_angle= ang + math.radians(deg)
        start_x = coord_x
        start_y = coord_y
        final_x = coord_x
        final_y = coord_y
        while 0 < final_x < self.map.shape[1] and 0 < final_y < self.map.shape[0] and abs(self.map[final_y, final_x]) < 0.0000001:
            start_x += 2 * np.cos(final_angle)
            start_y += 2 * np.sin(final_angle)
            final_x = int(round(start_x))
            final_y = int(round(start_y))
        end_p = np.array([final_x,final_y])
        start_p = np.array([coord_x,coord_y])
        dist = np.linalg.norm(end_p-start_p) * 10
        return dist

if __name__=='__main__':
    pass
