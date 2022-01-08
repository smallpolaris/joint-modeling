from __future__ import division
from scipy.interpolate import BSpline
import numpy as np
import math
import time


class Tools(object):
    def __init__(self, N, NT):
        self.N = N
        self.NT = NT



    def find_obs_num(self, y, l):
        rep_y = np.tile(y[:, np.newaxis], (1,l.shape[0]))
        ind = rep_y >= np.tile(l, (y.shape[0], 1))
        return np.sum(ind, axis=1)


    # def large_square(self, y, l=100):
    #     # Args: y is p * n matrix, output: y' * y; l is the segmentation arg ( seg y to l parts); large l is memory-friendly
    #     # p is too large
    #     lp = int(y.shape[0] / l)         # each y_l is lp * n
    #     y_square = np.zeros([y.shape[1], y.shape[1]])
    #     for i in range(l):
    #         y_l = y[lp * i:lp * (i+1)]
    #         y_square += np.matmul(y_l.transpose(), y_l)
    #     return y_square


    def large_square(self, y, l=100):
        # Args: y is p * n matrix, output: y' * y; l is the segmentation arg ( seg y to l parts); large l is memory-friendly
        # p is too large
        lp = int(y.shape[0] / l)         # each y_l is lp * n
        y_square = np.zeros([y.shape[1], y.shape[1]])
        for i in range(l):
            y_l = y[lp * i:lp * (i+1)]
            y_square += np.matmul(y_l.transpose(), y_l)
        return y_square



    def banded(self, g, N):
        """Creates a `g` generated banded matrix with 'N' rows"""
        n = len(g)
        T = np.zeros((N, N + n - 1))
        for x in range(N):
            T[x][x:x + n] = g
        return T




    #----------------------for analysis-------------------------------------------------
    def hrep_rmse(self, estimated, true, burnin):  # for parameter
        estimated = estimated[:, burnin:, ]
        rmse = np.sqrt(np.sum(np.power(estimated - true[np.newaxis, np.newaxis, :], 2), axis=1) / estimated.shape[1])
        # return np.mean(rmse, 0)
        return rmse


    def hrep_sd(self, estimated, burnin):    # for parameter
        estimated = estimated[:, burnin:, ]
        mean = np.mean(estimated[:, burnin:],1)
        sd = np.sqrt(np.sum(np.power(estimated - mean[:, np.newaxis], 2), axis=1) / estimated.shape[1])
        return np.mean(sd, 0)

    def rep_rmse(self, estimated, true, burnin):  # for effect
        estimated = estimated[:, burnin:, ]
        rmse = np.sqrt(np.sum(np.power(estimated - true[:, np.newaxis,], 2), axis=1) / estimated.shape[1])
        return rmse


    # def rep_sd(self, estimated, burnin):    # for effect
    #     estimated = estimated[:, burnin:, ]
    #     mean = np.mean(estimated[:, burnin:],1)
    #     sd = np.sqrt(np.sum(np.power(estimated - mean[:, np.newaxis], 2), axis=1) / estimated.shape[1])
    #     return np.mean(sd, 0)

    def cov_rate(self, t_value, p_value):
        ##------------------------t_value : shape:Rep------------------------
        ##------------------------p_value: shape: Rep * 3  # 0: true; 1: 5%percent; 2: 95% percent--------------------------##
        count = 0
        for i in range(p_value.shape[1]):
            if t_value[i] <= p_value[2, i] and t_value[i] >= p_value[1, i]:
                count += 1
        return count / t_value.shape[0]


    def hcov_rate(self, t_value, p_value):  # for vector
        ##------------------------t_value : shape:Rep------------------------
        ##------------------------p_value: shape: Rep * 3  # 0: true; 1: 5%percent; 2: 95% percent--------------------------##
        count = np.zeros_like(t_value)
        for j in range(t_value.shape[0]):
            for i in range(p_value.shape[1]):
                if t_value[j] <= p_value[1, i, j] and t_value[j] >= p_value[0, i, j]:
                    count[j] += 1
        return count / p_value.shape[1]


