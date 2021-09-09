import sys
import numpy as np
import math

class MotionModel:

    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 5]
    """

    def __init__(self):

        """
        TODO : Initialize Motion Model parameters here
        """

        self.alpha_1 = 0.0001
        self.alpha_2 = 0.0001
        self.alpha_3 = 0.01
        self.alpha_4 = 0.01


    def update(self, u_t0, u_t1, x_t0):
        """
        param[in] u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]   
        param[in] u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
        param[in] x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
        param[out] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        """

        """
        TODO : Fill in the '???' parts. 
        """

        # del_rot1 = ???
        # del_trans = ???
        # del_rot2 = ???

        # del_rot1_h = ??? - np.random.normal(???)
        # del_trans_h = ??? - np.random.normal(???)
        # del_rot2_h = ??? - np.random.normal(???)

        # x_t1 = ???

        # return x_t1

if __name__=="__main__":
    pass



