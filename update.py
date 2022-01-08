from __future__ import division
import numpy as np
import math
import tool
import time
from numpy.linalg import cholesky
from scipy.stats import invwishart, multivariate_normal
from scipy.interpolate import BSpline
import scipy.stats as ss
import time
# import torch
import numpy.random as nrd



class MCMC(tool.Tools):

    def __init__(self, N, NT, NX, NY, NW, ND, NF, NTRA, NETA, NG, O, OBS_T, t_set, NK, NB, K, knots):
        self.N = N
        self.NT = NT
        self.NX = NX   # order
        self.NY = NY
        self.NW = NW
        self.ND = ND
        self.NF = NF
        self.NTRA = NTRA
        self.NETA = NETA
        self.NG = NG
        self.O = O  # N * NT
        self.OBS_T = OBS_T   # OT is the observed longitudinal time
        # self.M = M
        self.t_set = t_set
        self.NK = NK
        self.NB = NB
        self.K = K
        self.knots = knots


    def update_lam_phi(self, C, y, phi, lam, Ind, x, xi, w, bs):
        #----------------------------Simulate eta----------------------------
        eta = np.zeros(shape=[self.N, self.NT, self.NETA])
        new_x = np.concatenate((x, xi), axis=2)  # N * NT * dim
        t_design = np.zeros(shape=[2, self.N, self.NT])
        for c in range(2):
            sp = BSpline(self.knots, bs[c], self.K)
            t_design[c] = sp(self.t_set) # N * NT
        # ran = t_design - np.sum(bs * self.b_igt, 1)[:, np.newaxis, np.newaxis]  # N * NT
        ran = t_design - np.mean(t_design, (1, 2))[:, np.newaxis, np.newaxis]
        obs = np.where(self.O == 0)
        eta[obs[0], obs[1]] = 0
        alpha_s0 = 9  # prior
        beta_s0 = 4
        for c in range(2):
            sample = np.where(C == c)
            # x_sample = new_x[sample]  # n * NT
            x_tra = w[sample[0], np.newaxis]
            eta = x_tra + ran[c, sample[0]]  # n * NT
            # to_add = np.ones([sample[0].shape[0], self.NT, 1])
            # eta_sample = np.concatenate((to_add, eta[:, :, np.newaxis]), 2)
            eta_sample = np.concatenate((xi[sample[0]], x[sample[0]], eta[:, :, np.newaxis]), 2)
            O_sample = self.O[sample]
            y_sample = y[sample]   #  this is for t_design  n * NT * NY
            obs = np.where(O_sample == 1)
            oeta = eta_sample[obs[0], obs[1]] # (NUM * DIM)
            oy = y_sample[obs[0], obs[1]]
            for k in range(y.shape[-1]):
                fix = np.where(Ind[k, :] == 0)
                free = np.where(Ind[k, :] == 1)
                y_star = oy[:, k] - np.sum(lam[c, k, fix[0]] * oeta[:, fix[0]], axis=1)
                U = np.transpose(oeta[:, free[0]])  # free * N
                H_0 = np.eye(free[0].shape[0]) * 4  # prior for free lam[k,]  free * free
                lam_0 = np.zeros(shape=[free[0].shape[0], 1])  # prior for free lam[k,]    free * 1
                temp = np.linalg.inv(H_0) + np.dot(U, np.transpose(U))  # free * free
                A = np.linalg.inv(temp)
                temp1 = np.dot(np.linalg.inv(H_0), lam_0) + np.dot(U, y_star[:, np.newaxis])  # free * 1
                a = np.dot(A, temp1)  # free * 1
                # for psi
                alpha_s = len(obs[0]) / 2 + alpha_s0
                temp2 = np.sum(y_star * y_star)
                temp3 = np.dot(np.dot(np.transpose(a), temp), a)
                temp4 = np.dot(np.dot(np.transpose(lam_0), np.linalg.inv(H_0)), lam_0)
                beta_s = beta_s0 + 1 / 2 * (temp2 - temp3 + temp4)
                phi[c, k] = 1 / nrd.gamma(alpha_s, 1 / beta_s[0, 0])
                if free[0].shape[0] == 1:
                    lam[c, k, free[0]] = nrd.normal(a[0, 0], np.sqrt(phi[c, k] * A[0, 0]))
                else:
                    lam[c, k, free[0]] = nrd.multivariate_normal(np.squeeze(a), phi[c, k] * A)
                # print("psi0:%.3f"%psi[s, 0])
        return lam, phi



    def update_trajectory(self, C, y, x, xi, tra, w, bs, lam, phi, c_tra, accept_tra):
        out = tra.copy()
        tra_star = np.zeros_like(tra)
        for c in range(2):
            tra_star[c] = ss.multivariate_normal.rvs(tra[c], c_tra[c] * np.identity(tra.shape[1]))
        y_likeli = self.y_likeli(y, x, xi, tra, w, bs, lam, phi)  # 2 * N * NT * NY
        y_likeli_star = self.y_likeli(y, x, xi, tra_star, w, bs, lam, phi)  # 2 * N * NT * NY
        for c in range(2):
            sample = np.where(C == c)
            p = np.sum(y_likeli[c, sample[0]])
            p_star = np.sum(y_likeli_star[c, sample[0]])
            pro = ss.multivariate_normal.logpdf(tra[c], mean=np.zeros_like(tra[c]), cov=100* np.identity(tra.shape[1]))
            pro_star = ss.multivariate_normal.logpdf(tra_star[c], mean=np.zeros_like(tra[c]), cov=100 * np.identity(tra.shape[1]))
            rand_ratio = nrd.uniform(0, 1)
            ratio_p = p_star + pro_star - p - pro
            ratio = np.exp(ratio_p) if ratio_p < 0 else 1
            if ratio > rand_ratio:
                out[c] = tra_star[c]
                accept_tra[c] += 1
        return out, accept_tra

    def update_bs(self, C, y, x, xi, w, bs, lam, phi, tau, c_bs, accept_bs):
        out = bs.copy()
        bs_star = np.zeros_like(bs)
        for c in range(2):
            bs_star[c] = ss.multivariate_normal.rvs(bs[c], c_bs[c] * np.identity(bs.shape[1]))
        eta = self.update_eta(bs, w)
        eta_star = self.update_eta(bs_star, w)
        y_likeli = self.y_likeli(y, x, xi, eta, lam, phi)  # 2 * N * NT * NY
        y_likeli_star = self.y_likeli(y, x, xi, eta_star, lam, phi)  # 2 * N * NT * NY
        D = self.banded([1, -2, 1], self.NB - 2)
        K = np.matmul(np.transpose(D), D)
        for c in range(2):
            sample = np.where(C == c)
            p = np.sum(y_likeli[c, sample[0]])
            p_star = np.sum(y_likeli_star[c, sample[0]])
            penalty = - 0.5 / tau[c] * np.matmul(np.matmul(bs[c, np.newaxis, :], K), bs[c, :, np.newaxis])
            penalty_star = - 0.5 /tau[c] * np.matmul(np.matmul(bs_star[c, np.newaxis, :], K), bs_star[c, :, np.newaxis])
            rand_ratio = nrd.uniform(0, 1)
            ratio_p = p_star + penalty_star - p - penalty
            ratio = np.exp(ratio_p) if ratio_p < 0 else 1
            if ratio > rand_ratio:
                out[c] = bs_star[c]
                accept_bs[c] += 1
        return out, accept_bs

    def update_tau(self, bs):
        a = 0.0001
        b = 0.0001
        r = self.NB - 2   # degree of k
        a_star = a + 0.5 * r
        D = self.banded([1, -2, 1], self.NB-2)
        K = np.matmul(np.transpose(D), D)
        b_star = b + 0.5 * np.matmul(np.matmul(bs[:, np.newaxis, :], K), bs[:, :, np.newaxis])[:, 0, 0]
        tau = 1/nrd.gamma(a_star, 1/b_star)
        return tau


    def update_sigma(self, C, eta, x, xi, w, tra):
        out = np.zeros(shape=[2])
        new_x = np.concatenate(
            (x, np.repeat(xi[:, np.newaxis], self.NT, axis=1)), axis=2)  # N * NT * dim
        nw = np.repeat(w[:, np.newaxis], self.NT, axis=1)  # N*NT*NW
        ntdesign = np.repeat(self.t_design[np.newaxis], self.N, axis=0)# N * NT * NW
        for c in range(2):
            sample = np.where(C == c)
            x_sample = new_x[sample]  # n * NT
            eta_sample = eta[sample] # w and b are not time-dependent
            w_sample = nw[sample]
            O_sample = self.O[sample]
            td_sample = ntdesign[sample]  # this is for t_design
            obs = np.where(O_sample == 1)
            ox = x_sample[obs[0], obs[1]]  # (NUM * DIM)
            otd = td_sample[obs[0], obs[1]]
            oeta = eta_sample[obs[0], obs[1]]
            ow = w_sample[obs[0], obs[1]]
            td_w = np.sum(otd * ow, axis=1)  # NUM
            eta_x = oeta[:, 0] - np.sum(ox * tra[c], 1) - td_w
            beta_s = 4 + 1 / 2 * np.sum(np.power(eta_x, 2))
            alpha_s = len(obs[0]) / 2 + 7  # Prior II
            # print("s:%d, alpha_s:%.6f"%(s,len(sample[0])))
            out[c] = 1 / np.random.gamma(alpha_s, 1 / beta_s, 1)[0]
        return out


    # def update_eta(self, C, x, xi, w, tra, sigma, y, Lam, phi):
    #     eta = np.zeros(shape=[self.N, self.NT, self.NETA])
    #     new_x = np.concatenate(
    #         (x, np.repeat(xi[:, np.newaxis], self.NT, axis=1)), axis=2)  # N * NT * dim
    #     ran = np.sum(self.t_design[np.newaxis] * w[:, np.newaxis], 2)  # N * NT
    #     mu = Lam[:, :, 0]
    #     lam = Lam[:, :, 1:]
    #     for c in range(2):
    #         sample = np.where(C == c)
    #         x_sample = new_x[sample]  # n * NT
    #         y_sample = y[sample] # n * NT * NY
    #         x_tra = np.sum(x_sample * tra[c], 2)
    #         m = x_tra + ran[sample]  # n * NT
    #         sigma_star = 1 / sigma[c] + np.matmul(np.matmul(lam[c].transpose(), np.diag(1/phi[c])), lam[c])[0, 0]
    #         m1 = 1/ sigma_star * np.matmul(np.matmul(lam[c].transpose(), np.diag(1/phi[c])), (y_sample-mu[c])[:, :, :, np.newaxis])# n * NT
    #         m2 = 1/ sigma_star * 1/sigma[c] * m
    #         eta[sample, :, 0] = nrd.normal(m1[:, :, 0, 0] + m2, np.sqrt(1/sigma_star))
    #     obs = np.where(self.O == 0)
    #     eta[obs[0], obs[1]] = 0
    #     return eta

    def update_w(self, C, xi, y, lam, phi, x, f, sur, OT, lambd, nu, grid, delta, w, bs, Sigma, c_w,
                 accept_w):
        cure = np.where(C == 1)
        nocure = np.where(C == 0)
        # w_star = np.zeros_like(w)
        # w_star[cure[0]] = w[cure[0]] + ss.multivariate_normal.rvs(cov=c_w[1] * np.identity(self.NW), size=cure[0].shape[0])
        # w_star[nocure[0]] = w[nocure[0]] + ss.multivariate_normal.rvs(cov=c_w[0] * np.identity(self.NW), size=nocure[0].shape[0])
        # w_star = w + ss.multivariate_normal.rvs(mean = np.zeros([self.NW]),  cov=c_w * np.identity(self.NW), size=self.N)
        w_star = w + nrd.normal(0, c_w, size=self.N)
        ###--------------------------------------Incidence model-------------------------------------
        # n_d = np.concatenate((d, xi[:, 0], w[:, np.newaxis]), 1)
        # m = np.exp(np.sum(n_d * inc, 1))
        p_inc = np.zeros(shape=[self.N])
        # p_inc[cure] = (m / (1+m))[cure]
        # p_inc[nocure] = (1/ (1+m))[nocure]
        # n_dstar = np.concatenate((d, xi[:, 0], w_star[:, np.newaxis]), 1)
        # m_star = np.exp(np.sum(n_dstar * inc, 1))
        pinc_star = np.zeros(shape=[self.N])
        # pinc_star[cure] = (m_star/(1+m_star))[cure]
        # pinc_star[nocure] = 1/(1+m_star)[nocure]
        # ------------------------------------Factor analysis model----------------------------------#
        ptra = np.zeros([self.N])
        ptra_star = np.zeros([self.N])
        eta = self.update_eta(bs, w)
        eta_star = self.update_eta(bs, w_star)
        y_likeli = np.sum(self.y_likeli(y, x, xi, eta, lam, phi), (2, 3))
        y_likeli_star = np.sum(self.y_likeli(y, x, xi, eta_star, lam, phi), (2, 3))
        ptra[cure[0]] = y_likeli[1, cure[0]]
        ptra[nocure[0]] = y_likeli[0, nocure[0]]
        ptra_star[cure[0]] = y_likeli_star[1, cure[0]]
        ptra_star[nocure[0]] = y_likeli_star[0, nocure[0]]
        # ------------------------------Survival model---------------------------
        sur_coeff = sur[-self.NW:]
        psur = self.sur_prob(f, xi, w, sur, OT, lambd, nu, grid)
        psur_star = self.sur_prob(f, xi, w_star, sur, OT, lambd, nu, grid)
        event = np.where(delta == 1)  # happened data
        psur[event] = psur[event] + sur_coeff * w[event[0]]
        psur_star[event] = psur_star[event] + sur_coeff * w_star[event]
        psur[cure[0]] = 0
        psur_star[cure[0]] = 0
        # psur = 0
        # psur_star = 0
        # ------------------------------Variance--------------------
        p = np.zeros(shape=[self.N])
        p_star = np.zeros(shape=[self.N])
        p[cure[0]] = ss.norm.logpdf(w[cure[0]], loc=np.zeros(self.NW), scale=np.sqrt(Sigma[1]))
        p_star[cure[0]] = ss.norm.logpdf(w_star[cure[0]], loc=np.zeros(self.NW), scale=np.sqrt(Sigma[1]))
        p[nocure[0]] = ss.norm.logpdf(w[nocure[0]], loc=np.zeros(self.NW), scale=np.sqrt(Sigma[0]))
        p_star[nocure[0]] = ss.norm.logpdf(w_star[nocure[0]], loc=np.zeros(self.NW), scale=np.sqrt(Sigma[0]))
        log_ratio = pinc_star + ptra_star + psur_star + p_star - p_inc - ptra - psur - p
        ratio = np.ones([self.N])
        ll = np.where(log_ratio < 0)
        ratio[ll] = np.exp(log_ratio[ll])
        rand_ratio = nrd.uniform(0, 1, self.N)
        acc = np.where(ratio > rand_ratio)
        w[acc] = w_star[acc]
        accept_w[acc] += 1
        return w, accept_w


    def update_eta(self, bs, w):
        mean = w[:, np.newaxis] #2 * N * NT
        #-------------------------Calculate t_design----------------------
        t_design = np.zeros([2, self.N, self.NT])
        for c in range(2):
            sp = BSpline(self.knots, bs[c], self.K)
            t_design[c] = sp(self.t_set) # N * NT
        # ran = t_design - np.sum(bs * self.b_igt, 1)[:, np.newaxis, np.newaxis]# 2 * N * NT
        ran = t_design - np.mean(t_design, (1, 2))[:, np.newaxis, np.newaxis]  # 2 * N * NT
        eta = mean + ran  # 2* N * NT
        return eta[:, :, :, np.newaxis]

    def y_likeli(self, y, x, xi, eta, lam, phi):
        #Args: Return the logpdf of trajectory in two group adn set y_likeli = 0 for unobserved time
        y_likeli = np.zeros([2, self.N, self.NT, self.NY])
        # to_add = np.ones([2, self.N, self.NT, 1])
        # new_x = np.concatenate((x, np.repeat(xi[:, np.newaxis], self.NT, axis=1)), axis=2)  # N * NT * dim
        # new_x = np.concatenate((x, xi), axis=2)[:, :, :, np.newaxis]  # N * NT * dim
        # n_eta = np.concatenate((x[np.newaxis], eta[:, :, :, np.newaxis]), axis=3) # 2 * N * NT * dim
        Lam = lam
        # new_x = np.concatenate((x, xi, n_eta), axis=2)
        obs = np.where(self.O == 0)
        for c in range(2):
            m = np.matmul(Lam[c], np.concatenate((xi, x, eta[c]), axis=2)[:, :, :, np.newaxis]) #N * NT * 3 * 1
            for ny in range(self.NY):
                y_likeli[c, :, :, ny] = ss.norm.logpdf(y[:, :, ny], m[:, :, ny, 0], np.sqrt(phi[c, ny]))
        y_likeli[:, obs[0], obs[1]] = 0
        return y_likeli


    def ty_likeli(self, y, x, xi, tra, w, bs, lam, phi):
        #Args: Return the logpdf of trajectory in two group adn set y_likeli = 0 for unobserved time
        y_likeli = np.zeros([2, self.N, self.NT, self.NY])
        to_add = np.ones([2, self.N, self.NT, 1])
        new_x = np.concatenate((x, np.repeat(xi[:, np.newaxis], self.NT, axis=1), np.repeat(w[:, np.newaxis, np.newaxis], self.NT, axis=1)), axis=2)
        mean = np.sum(new_x * tra[:, np.newaxis, np.newaxis], 3)  #2 * N * NT
        #-------------------------Calculate t_design----------------------
        t_design = np.zeros([2, self.N, self.NT])
        # for c in range(2):
        #     sp = BSpline(self.knots, bs[c], self.K)
        #     t_design[c] = sp(self.t_set) # N * NT
        t_design[0] = np.sqrt(self.t_set)  # This is f(t) for nocured group
        t_design[1] = np.log(self.t_set + 1)
        t_design = t_design - np.mean(t_design, (1, 2))[:, np.newaxis, np.newaxis]
        ran = t_design # 2 * N * NT
        eta = mean + ran
        n_eta = np.concatenate((to_add, eta[:, :, :, np.newaxis]), axis=3) # 2 * N * NT * dim
        obs = np.where(self.O == 0)
        for c in range(2):
            m = np.matmul(lam[c], n_eta[c, :, :, :, np.newaxis]) #N * NT * 3 * 1
            for ny in range(self.NY):
                y_likeli[c, :, :, ny] = ss.norm.logpdf(y[:, :, ny], m[:, :, ny, 0], np.sqrt(phi[c, ny]))
        y_likeli[:, obs[0], obs[1]] = 0
        return y_likeli

    def w_likeli(self, w, Sigma):
        #Args: Return the logpdf of random effect in two group
        w_likeli = np.zeros([2, self.N])
        w_likeli[0] = ss.multivariate_normal.logpdf(w, mean=np.zeros(self.NW), cov=Sigma[0])
        w_likeli[1] = ss.multivariate_normal.logpdf(w, mean=np.zeros(self.NW), cov=Sigma[1])
        return w_likeli


    # def update_w(self, C, eta, x, xi, tra, sigma, f, sur, OT, lambd, nu, grid, delta, w, Sigma, c_w, accept_w):
    #     cure = np.where(C == 1)
    #     nocure = np.where(C == 0)
    #     # design_square = np.matmul(self.t_design[:, :, np.newaxis], self.t_design[:, np.newaxis])
    #     # prior = np.zeros(shape=[self.N, self.NW, self.NW])
    #     # sum_design_square = np.cumsum(design_square, axis=0)
    #     # for t in range(design_square.shape[0]):
    #     #     sam = np.where(self.OBS_T == t+1)
    #     #     prior[sam] = sum_design_square[t]
    #     # tg_sigma = np.eye(self.NW) * 100  # proposal for w
    #     eta_likeli = self.y_likeli(eta[:, :, 0], x, xi, w, tra, sigma)
    #     w_star = w.copy()
    #     # w_zero = w.copy()
    #     sur_coeff = sur[-self.NW:]
    #     #=============================Calculate some Fisher information======================
    #     # prior[cure] = prior[cure] / phi[ny, 1]
    #     # prior[nocure] = prior[nocure] / phi[ny, 0]
    #     # w_zero[:, ny * self.NW:(ny+1) * self.NW] = 0
    #     # sur_prob = self.sur_prob(f, xi, w_zero, b, sur, OT, lambd, nu, grid)   # Need to notice this is to evaluate at w = 0
    #     # prior[nocure] += -sur_prob[nocure[0], np.newaxis, np.newaxis] * np.matmul(sur_coeff[:, np.newaxis], sur_coeff[np.newaxis])
    #     # sur_sigma = np.linalg.inv(prior + np.linalg.inv(tg_sigma))
    #     # w_pro = ss.multivariate_normal.rvs(mean=w[:, ny * self.NW : (ny+1)*self.NW, np.newaxis], cov=c_w * sur_sigma)
    #     # w_star[ny * self.NW:(ny+1)*self.NW] = w_pro
    #     #----------------------------Dont use fisher information----------------------------------
    #     # sur_sigma = np.eye(self.NW)
    #     # for i in range(self.N):
    #     #     w_star[i, ny * self.NW:(ny + 1) * self.NW] = ss.multivariate_normal.rvs(mean=w[i, ny * self.NW:(ny+1)*self.NW], cov=c_w[ny] * sur_sigma)
    #     # w_star[:, ny * self.NW:(ny + 1) * self.NW] = w[:, ny * self.NW:(ny + 1) * self.NW] +\
    #     #                                              ss.multivariate_normal.rvs(cov=c_w[ny] * sur_sigma, size=self.N)
    #     w_star[cure[0]] = w[cure[0]] + ss.multivariate_normal.rvs(cov=c_w * Sigma[1], size=cure[0].shape[0])
    #     w_star[nocure[0]] = w[nocure[0]] + ss.multivariate_normal.rvs(cov=c_w * Sigma[0], size=nocure[0].shape[0])
    #     # ---------------------------------Trajectory model-------------------------
    #     ptra = np.zeros([self.N])
    #     ptra[nocure] = np.sum(eta_likeli[0, nocure[0]], 1)
    #     ptra[cure] = np.sum(eta_likeli[1, cure[0]], 1)
    #     ptra_star = np.zeros([self.N])
    #     eta_likelistar = self.y_likeli(eta[:, :, 0], x, xi, w_star, tra, sigma)
    #     ptra_star[nocure] = np.sum(eta_likelistar[0, nocure[0]], 1)
    #     ptra_star[cure] = np.sum(eta_likelistar[1, cure[0]], 1)
    #     # ------------------------------Survival model---------------------------
    #     psur = self.sur_prob(f, xi, w, sur, OT, lambd, nu, grid)
    #     psur_star = self.sur_prob(f, xi, w_star, sur, OT, lambd, nu, grid)
    #     event = np.where(delta == 1)  # happened data
    #     psur[event] = psur[event] + np.sum(sur_coeff * w[event[0]], 1)
    #     psur_star[event] = psur_star[event] + np.sum(sur_coeff * w_star[event[0]], 1)
    #     psur[cure[0]] = 0
    #     psur_star[cure[0]] = 0
    #     # psur = 0
    #     # psur_star = 0
    #     # ------------------------------Variance--------------------
    #     p = np.zeros(shape=[self.N])
    #     p_star = np.zeros(shape=[self.N])
    #     p[cure] = ss.multivariate_normal.logpdf(w[cure], mean=np.zeros(self.NW), cov=Sigma[1])
    #     p[nocure] = ss.multivariate_normal.logpdf(w[nocure], mean=np.zeros(self.NW), cov=Sigma[0])
    #     p_star[cure] = ss.multivariate_normal.logpdf(w_star[cure], mean=np.zeros(self.NW), cov=Sigma[1])
    #     p_star[nocure] = ss.multivariate_normal.logpdf(w_star[nocure], mean=np.zeros(self.NW), cov=Sigma[0])
    #     log_ratio = ptra_star + psur_star + p_star - ptra - psur - p
    #     ratio = np.ones([self.N])
    #     ll = np.where(log_ratio < 0)
    #     ratio[ll] = np.exp(log_ratio[ll])
    #     rand_ratio = nrd.uniform(0, 1, self.N)
    #     acc = np.where(ratio > rand_ratio)
    #     w[acc] = w_star[acc]
    #     accept_w[acc] += 1
    #     return w, accept_w




    # def update_trajectory(self, y, x, xi, w, C, b, phi):
    #     out = np.zeros(shape=[2, self.NY, self.NTRA])
    #     new_x = np.concatenate((x, np.repeat(xi[:, np.newaxis], self.NT, axis=1), np.repeat(self.t_design[np.newaxis], self.N, 0),
    #                             np.repeat(b[:, np.newaxis], self.NT, axis=1)), axis=2) # N * NT * dim
    #     # new_x = np.concatenate((x, np.repeat(xi[:, np.newaxis], self.NT, axis=1),
    #     #                         np.repeat(b[:, np.newaxis], self.NT, axis=1)), axis=2) # N * NT * dim   !!!!!!!!!!!!!!!!Debug!!!!!!!!!
    #     nw = np.repeat(w[:, np.newaxis], self.NT, axis=1) # N*NT*(NY*NW)
    #     # nb = np.repeat(b[:, np.newaxis], self.NT, axis=1)
    #     ntdesign = np.repeat(self.t_design[np.newaxis], self.N, axis=0)
    #     for c in range(2):
    #         sample = np.where(C == c)
    #         x_sample = new_x[sample] # n * NT
    #         # b_sample = nb[sample] # w and b are not time-dependent
    #         O_sample = self.O[sample]
    #         td_sample = ntdesign[sample]   #  this is for t_design
    #         obs = np.where(O_sample == 1)
    #         ox = x_sample[obs[0], obs[1]] # (NUM * DIM)
    #         otd = td_sample[obs[0], obs[1]]
    #         x_sum = np.matmul(np.transpose(ox), ox)
    #         for ny in range(self.NY):
    #             y_sample = y[sample[0], :, ny]
    #             w_sample = nw[sample[0], :, ny * self.NW:(ny+1) * self.NW]
    #             oy = y_sample[obs[0], obs[1]]
    #             ow = w_sample[obs[0], obs[1]] # NUM * NW
    #             # ob = b_sample[obs[0], obs[1]] # NUM * 1
    #             # print("x_sum:")
    #             # print(x_sum)
    #             td_w = np.sum(otd * ow, axis=1)  # NUM
    #             x_y = np.sum(np.transpose(ox) * (oy - td_w), 1)
    #             # print("x_y:")
    #             # print(x_y)
    #             sigma_tra = np.linalg.inv(x_sum /phi[c, ny] + np.diag(v=np.ones(shape=x_sum.shape[0])) / 100)  # 100： prior
    #             miu_tra = np.squeeze(np.dot(sigma_tra, x_y[:, np.newaxis]) / phi[c, ny], 1)
    #             # print("sigma_tra, miu_tra:")
    #             # print(sigma_tra)
    #             # print(miu_tra)
    #             tra_star = nrd.multivariate_normal(miu_tra, sigma_tra)
    #             out[c, ny] = tra_star
    #     return out

    def update_phi(self, y, x, xi, w, C, b, tra):
        out = np.zeros(shape=[2, self.NY])
        new_x = np.concatenate((x, np.repeat(xi[:, np.newaxis], self.NT, axis=1), np.repeat(self.t_design[np.newaxis], self.N, 0),
                                np.repeat(b[:, np.newaxis], self.NT, axis=1)), axis=2) # N * NT * dim
        nw = np.repeat(w[:, np.newaxis], self.NT, axis=1)
        # nb = np.repeat(b[:, np.newaxis], self.NT, axis=1)
        ntdesign = np.repeat(self.t_design[np.newaxis], self.N, axis=0)
        for c in range(2):
            sample = np.where(C == c)
            x_sample = new_x[sample] # n * NT
            # b_sample = nb[sample] # w and b are not time-dependent
            O_sample = self.O[sample]
            td_sample = ntdesign[sample]   #  this is for t_design
            obs = np.where(O_sample == 1)
            ox = x_sample[obs[0], obs[1]] # (NUM * DIM)
            otd = td_sample[obs[0], obs[1]]
            for ny in range(self.NY):
                y_sample = y[sample[0], :, ny]
                w_sample= nw[sample[0], :, ny*self.NW:(ny+1)*self.NW]
                oy = y_sample[obs[0], obs[1]]
                ow = w_sample[obs[0], obs[1]] # NUM * NW
            # ob = b_sample[obs[0], obs[1]] # NUM * 1
                td_w = np.sum(otd * ow, axis=1)  # NUM
                y_x = oy - np.sum(ox * tra[c, ny], 1) - td_w
                beta_s = 4 + 1 / 2 * np.sum(np.power(y_x, 2))
                alpha_s = len(obs[0]) / 2 + 7  # Prior II
            # print("s:%d, alpha_s:%.6f"%(s,len(sample[0])))
                out[c, ny] = 1 / np.random.gamma(alpha_s, 1/beta_s, 1)[0]
        return out



    def w_likeli(self, w, Sigma):
        #Args: return the loglikeli  of random effect
        w_likeli = np.zeros([2, self.N])
        w_likeli[0] = ss.norm.logpdf(w, loc=0, scale=np.sqrt(Sigma[0]))
        w_likeli[1] = ss.norm.logpdf(w, loc=0, scale=np.sqrt(Sigma[1]))
        return w_likeli

    def pc(self, d, xi, w, inc):
        #return the pdf of logistic regression
        nd = np.concatenate((d, xi[:, 0]), 1) # N * dim
        mm = np.exp(np.sum(nd * inc, 1)) # N
        pc = mm / (1 + mm)
        return pc
        #Args: return the log prob of logistic incidence model

    def sur_prob(self, f, xi, w, sur, OT, lambd, nu, grid):
        #Args: Return the log survival prbability of not happenend data#
        const = 0
        for g in range(self.NG):
            v_1 = lambd[g] * (OT - grid[g])
            v_2 = 0
            if g != 0:
                for j in range(g):
                    v_2 += lambd[j] * (grid[j + 1] - grid[j])
            const += nu[:, g] * (v_1 + v_2)  # the integral of \lambda(t)
        n_c = np.concatenate((f, xi[:, 0], w[:, np.newaxis]), 1)
        cond = np.sum(sur * n_c, 1)
        sur = - const * np.exp(cond)
        return sur

    def update_C(self, y_likeli, sur_prob, p, sample, w_likeli):
        #Args: y_likeli: 2 * N * NT * NY, sur_prob: N, p: N: ss is sample of delta == 0
        # p(C=1) = p * np.exp(y_likeli[1] + w_likeli[1])
        # p (C = 0) = (1-p) * np.exp(sur_prob + y_likeli[0] + w_likeli[0])
        pc = p[sample] * np.exp(np.sum(y_likeli[1, sample[0]], (1, 2)) + w_likeli[1, sample[0]])
        pn = (1 - p[sample]) * np.exp(np.sum(y_likeli[0, sample[0]], (1, 2))  + w_likeli[0, sample[0]] + sur_prob[sample])
        sp = pc / (pc + pn)    # standardized cured prob
        C = np.zeros(self.N)
        C[sample] = nrd.binomial(1, sp)
        return C

    def update_y(self, y, x, xi, w, tra, C, b, phi, t_y, m_y):
        new_y = y.copy()
        new_x = np.concatenate((x, np.repeat(xi[:, np.newaxis], self.NT, axis=1), np.repeat(self.t_design[np.newaxis], self.N, 0),
                                np.repeat(b[:, np.newaxis], self.NT, axis=1)), axis=2) # N * NT * dim
        # new_x = np.concatenate((x, np.repeat(xi[:, np.newaxis], self.NT, axis=1),
        #                         np.repeat(b[:, np.newaxis], self.NT, axis=1)), axis=2) # N * NT * dim  for debug
        for c in range(2):
            sample = np.where(C == c)
            m = np.sum(new_x[sample] * tra[c], 2)  # N * NT   mean not added random effect w
            ran = np.sum(self.t_design[np.newaxis] * w[:, np.newaxis], 2)# N * NT
            mm = m + ran[sample]  # N * NT
            # aa = m_y[sample]
            y_c = y[sample]
            loc = np.where(y_c == 0)
            mean = mm[loc[0], loc[1]]
            # nn = ss.truncnorm.rvs(-999999999999999, (0-mean)/np.sqrt(phi[c]), loc=mean, scale=np.sqrt(phi[c]))
            nn = nrd.normal(mean, np.sqrt(phi[c]))
            y_c[loc[0], loc[1]] = nn
            new_y[sample] = y_c
            # ty_c = t_y[sample]
        return new_y

    # def update_b(self, C, d, xi, inc, y, x, phi, w, tra, f, sur, OT, lambd, nu, grid, delta, b, sigma, c_b, accept_b):
    #     b_out = b.copy()
    #     b_star = np.zeros_like(b)
    #     b_star[:, 0] = nrd.normal(b[:, 0], c_b)
    #     #-------------------------------Incidence model--------------------------
    #     pinc = np.zeros(shape=[self.N])
    #     n_d = np.concatenate((d, xi), axis=1)
    #     loc_d = np.where(C == 0)  # disease group
    #     loc_c = np.where(C == 1)  # cure group
    #     inc_d = np.exp(np.sum(inc * n_d, 1) + b[:, 0])  # N
    #     inc_star = np.exp(np.sum(inc * n_d, 1) + b_star[:, 0])
    #     pinc[loc_c[0]] = np.log(inc_d / (1 + inc_d))[loc_c[0]]
    #     pinc[loc_d[0]] = np.log(1 / (1 + inc_d))[loc_d[0]]
    #     pinc_star = np.zeros([self.N])
    #     pinc_star[loc_c[0]] = np.log(inc_star / (1 + inc_star))[loc_c[0]]
    #     pinc_star[loc_d[0]] = np.log(1 / (1 + inc_star))[loc_d[0]]
    #     #---------------------------------Trajectory model-------------------------
    #     y_likeli = self.y_likeli(y, x, xi, w, b, tra, phi)
    #     ptra = np.zeros([self.N])
    #     ptra[loc_d[0]] = np.sum(y_likeli[0, loc_d[0]], (1, 2))
    #     ptra[loc_c[0]] = np.sum(y_likeli[1, loc_c[0]], (1, 2))
    #     ptra_star = np.zeros([self.N])
    #     y_likelistar = self.y_likeli(y, x, xi, w, b_star, tra, phi)
    #     ptra_star[loc_d[0]] = np.sum(y_likelistar[0, loc_d[0]], (1, 2))
    #     ptra_star[loc_c[0]] = np.sum(y_likelistar[1, loc_c[0]], (1, 2))
    #     #------------------------------Survival model---------------------------
    #     psur = self.sur_prob(f, xi, w, b, sur, OT, lambd, nu, grid)
    #     psur_star = self.sur_prob(f, xi, w, b_star, sur, OT, lambd, nu, grid)
    #     event = np.where(delta == 1)  # happened dta
    #     psur[event[0]] = psur[event[0]] + sur[-1] * b[event[0], 0]
    #     psur_star[event[0]] = psur_star[event[0]] + sur[-1] * b_star[event[0], 0]
    #     psur[loc_c[0]] = 0
    #     psur_star[loc_c[0]] = 0
    #     #------------------------------Variance--------------------
    #     p = ss.norm.logpdf(b[:, 0], loc=np.zeros(self.N), scale=np.sqrt(sigma))
    #     p_star = ss.norm.logpdf(b_star[:, 0], loc=np.zeros(self.N), scale=np.sqrt(sigma))
    #     log_ratio = pinc_star + ptra_star + psur_star + p_star - pinc - ptra - psur - p
    #     ratio = np.ones([self.N])
    #     ll = np.where(log_ratio < 0)
    #     ratio[ll] = np.exp(log_ratio[ll])
    #     rand_ratio = nrd.uniform(0, 1, self.N)
    #     acc = np.where(ratio > rand_ratio)
    #     b_out[acc] = b_star[acc]
    #     accept_b[acc] += 1
    #     return b_out, accept_b


    # def update_w(self, C, xi, y, x, phi, b, tra, f, sur, OT, lambd, nu, grid, delta, w, Sigma, c_w, accept_w):
    #     cure = np.where(C == 1)
    #     nocure = np.where(C == 0)
    #     design_square = np.matmul(self.t_design[:, :, np.newaxis], self.t_design[:, np.newaxis])
    #     prior = np.zeros(shape=[self.N, self.NW, self.NW])
    #     sum_design_square = np.cumsum(design_square, axis=0)
    #     for t in range(design_square.shape[0]):
    #         sam = np.where(self.OBS_T == t+1)
    #         prior[sam] = sum_design_square[t]
    #     tg_sigma = np.eye(self.NW) * 100  # proposal for w
    #     y_likeli = self.y_likeli(y, x, xi, w, b, tra, phi)
    #     for ny in range(self.NY):
    #         w_star = w.copy()
    #         # w_zero = w.copy()
    #         sur_coeff = sur[f.shape[1] + xi.shape[1] + ny * self.NW:f.shape[1] + xi.shape[1] + (ny + 1) * self.NW]
    #         #=============================Calculate some Fisher information======================
    #         # prior[cure] = prior[cure] / phi[ny, 1]
    #         # prior[nocure] = prior[nocure] / phi[ny, 0]
    #         # w_zero[:, ny * self.NW:(ny+1) * self.NW] = 0
    #         # sur_prob = self.sur_prob(f, xi, w_zero, b, sur, OT, lambd, nu, grid)   # Need to notice this is to evaluate at w = 0
    #         # prior[nocure] += -sur_prob[nocure[0], np.newaxis, np.newaxis] * np.matmul(sur_coeff[:, np.newaxis], sur_coeff[np.newaxis])
    #         # sur_sigma = np.linalg.inv(prior + np.linalg.inv(tg_sigma))
    #         # w_pro = ss.multivariate_normal.rvs(mean=w[:, ny * self.NW : (ny+1)*self.NW, np.newaxis], cov=c_w * sur_sigma)
    #         # w_star[ny * self.NW:(ny+1)*self.NW] = w_pro
    #         #----------------------------Dont use fisher information----------------------------------
    #         sur_sigma = np.eye(self.NW)
    #         # for i in range(self.N):
    #         #     w_star[i, ny * self.NW:(ny + 1) * self.NW] = ss.multivariate_normal.rvs(mean=w[i, ny * self.NW:(ny+1)*self.NW], cov=c_w[ny] * sur_sigma)
    #         # w_star[:, ny * self.NW:(ny + 1) * self.NW] = w[:, ny * self.NW:(ny + 1) * self.NW] +\
    #         #                                              ss.multivariate_normal.rvs(cov=c_w[ny] * sur_sigma, size=self.N)
    #         w_star[cure[0], ny * self.NW:(ny + 1) * self.NW] = w[cure[0], ny * self.NW:(ny + 1) * self.NW] +\
    #                                                      ss.multivariate_normal.rvs(cov=c_w * Sigma[1, ny*self.NW:(ny+1)*self.NW, ny*self.NW:(ny+1)*self.NW], size=cure[0].shape[0])
    #         w_star[nocure[0], ny * self.NW:(ny + 1) * self.NW] = w[nocure[0], ny * self.NW:(ny + 1) * self.NW] +\
    #                                                      ss.multivariate_normal.rvs(cov=c_2 * Sigma[0, ny*self.NW:(ny+1)*self.NW, ny*self.NW:(ny+1)*self.NW], size=nocure[0].shape[0])
    #         # ---------------------------------Trajectory model-------------------------
    #         ptra = np.zeros([self.N])
    #         ptra[nocure] = np.sum(y_likeli[0, nocure[0], :, ny], 1)
    #         ptra[cure] = np.sum(y_likeli[1, cure[0], :, ny], 1)
    #         ptra_star = np.zeros([self.N])
    #         y_likelistar = self.y_likeli(y, x, xi, w_star, b, tra, phi)
    #         ptra_star[nocure] = np.sum(y_likelistar[0, nocure[0], :, ny], 1)
    #         ptra_star[cure] = np.sum(y_likelistar[1, cure[0], :, ny], 1)
    #         # ------------------------------Survival model---------------------------
    #         psur = self.sur_prob(f, xi, w, b, sur, OT, lambd, nu, grid)
    #         psur_star = self.sur_prob(f, xi, w_star, b, sur, OT, lambd, nu, grid)
    #         event = np.where(delta == 1)  # happened data
    #         psur[event] = psur[event] + np.sum(sur_coeff * w[event[0], ny * self.NW:(ny+1) * self.NW], 1)
    #         psur_star[event] = psur_star[event] + np.sum(sur_coeff * w_star[event[0], ny * self.NW:(ny+1) * self.NW], 1)
    #         psur[cure[0]] = 0
    #         psur_star[cure[0]] = 0
    #         # ------------------------------Variance--------------------
    #         p = np.zeros(shape=[self.N])
    #         p_star = np.zeros(shape=[self.N])
    #         p[cure] = ss.multivariate_normal.logpdf(w[cure], mean=np.zeros(self.NY * self.NW), cov=Sigma[1])
    #         p[nocure] = ss.multivariate_normal.logpdf(w[nocure], mean=np.zeros(self.NY * self.NW), cov=Sigma[0])
    #         p_star[cure] = ss.multivariate_normal.logpdf(w_star[cure], mean=np.zeros(self.NY * self.NW), cov=Sigma[1])
    #         p_star[nocure] = ss.multivariate_normal.logpdf(w_star[nocure], mean=np.zeros(self.NY * self.NW), cov=Sigma[0])
    #         log_ratio = ptra_star + psur_star + p_star - ptra - psur - p
    #         ratio = np.ones([self.N])
    #         ll = np.where(log_ratio < 0)
    #         ratio[ll] = np.exp(log_ratio[ll])
    #         rand_ratio = nrd.uniform(0, 1, self.N)
    #         acc = np.where(ratio > rand_ratio)
    #         w[acc] = w_star[acc]
    #         accept_w[acc, ny] += 1
    #     return w, accept_w




    def update_incidence(self, d, xi, w, C, inc, c_inc, accept_inc):
        ##=========================== update logistic incidence model=======================##
        # out = np.zeros_like(inc)
        #--------------------------------Fisher information-------------------------#
        n_d = np.concatenate((d, xi[:, 0]), axis=1)
        d_d = np.matmul(n_d[:, :, np.newaxis], n_d[:, np.newaxis]) # N * dim * dim
        # eb = np.exp(b) / np.square(1 + np.exp(b))
        pri_Sigma = np.eye(inc.shape[0])
        # Sigma = np.sum(d_d * eb[:, :, np.newaxis], 0) + np.linalg.inv(pri_Sigma)
        Sigma = np.sum(d_d / 4, 0) + np.linalg.inv(pri_Sigma)
        inc_star = nrd.multivariate_normal(inc, c_inc * np.linalg.inv(Sigma))
        #-----------------------Calculate ratio---------------------------------#
        loc_d = np.where(C == 0)  # disease group
        loc_c = np.where(C == 1)  # cure group
        inc_d = np.exp(np.sum(inc * n_d, 1))  # N
        p_c = np.log(inc_d / (1 + inc_d))[loc_c[0]]
        p_d = np.log(1 / (1 + inc_d))[loc_d[0]]
        pri_mean = np.zeros_like(inc) # prior
        p = np.sum(p_c) + np.sum(p_d) + multivariate_normal.logpdf(inc, pri_mean, pri_Sigma)
        star_d = np.exp(np.sum(inc_star * n_d, 1))   # N
        p_sc = np.log(star_d/(1+star_d))[loc_c[0]]
        p_sd = np.log(1/(1+star_d))[loc_d[0]]
        p_s = np.sum(p_sc) + np.sum(p_sd) + multivariate_normal.logpdf(inc_star, pri_mean, pri_Sigma)
        a = p_s - p
        ap = np.exp(a) if a<0 else 1
        if ap > nrd.rand(1):
            return inc_star, accept_inc+1
        else:
            return inc, accept_inc

    def update_lambd(self, lambd, nu, delta, sur, f, xi, w, OT, grid, C):
        alpha_1 = 0.2
        alpha_2 = 0.4
        sample = np.where(C == 0)
        n_f = np.concatenate((f, xi[:, 0], w[:, np.newaxis]), 1)
        f_s = n_f[sample]
        cond = np.exp(np.sum(sur * f_s, 1))
        OT_s = OT[sample]
        nu_s = nu[sample]
        delta_s = delta[sample]
        for g in range(self.NG):
            v_1 = nu_s[:, g] * (OT_s - grid[g])
            v_2 = 0
            if g != self.NG-1:
                for j in range(g + 1, self.NG):
                    v_2 += nu_s[:, j] * (grid[g+1] - grid[g])
                    # v_2 += nu[:, j] * (t[j + 1] - t[j])
            alpha_2_p = alpha_2 + np.sum(cond * (v_1 + v_2))
            alpha_1_p = np.sum(nu_s[:, g] * delta_s) + alpha_1
            lambd[g] = nrd.gamma(alpha_1_p, 1/alpha_2_p)
        return lambd

    def update_sur(self, sur, f, xi, w, delta, OT, nu, lambd, s, c_sur, accept_sur, C):
        tg_0 = 2 # prior for theta_gamma
        tg_sigma = np.eye(sur.shape[0]) * 100
        x = np.concatenate((f, xi[:, 0], w[:, np.newaxis]), 1) # incorporate all variables N * dim
        sample = np.where(C == 0)
        x_s = x[sample]
        OT_s = OT[sample]
        nu_s = nu[sample]
        delta_s = delta[sample]
        x_x = np.matmul(x_s[:, :, np.newaxis], x_s[:, np.newaxis, :])
        # lambda_s = 0
        const = 0
        v_3 = 0
        for g in range(self.NG):
            v_3 += nu_s[:, g] * delta_s
            v_1 = lambd[g] * (OT_s - s[g])
            v_2 = 0
            if g != 0:
                for j in range(g):
                    v_2 += lambd[j] * (s[j+1] - s[j])
            const += nu_s[:, g] * (v_1 + v_2)
        # const *= w[:, 0, 1]
        sur_sigma = np.linalg.inv(np.sum(const[:, np.newaxis, np.newaxis] * x_x, 0) + np.linalg.inv(tg_sigma))
        sur_star = nrd.multivariate_normal(sur, c_sur * sur_sigma)
        # theta_gamma_star = nrd.multivariate_normal(theta_gamma, c_theta_gamma * np.eye(theta_gamma.shape[0]))
        cond = np.sum(x_s * sur, 1)
        cond_star = np.sum(x_s * sur_star, 1)
        ratio_1 = v_3 * (cond_star - cond)
        ratio_2 = const * (np.exp(cond) - np.exp(cond_star))
        prior_ratio = - np.dot(np.dot((sur - tg_0)[np.newaxis, :], np.linalg.inv(tg_sigma)),
                           (sur - tg_0)[:, np.newaxis]).squeeze()
        prior_ratio_star = - np.dot(np.dot((sur_star - tg_0)[np.newaxis, :], np.linalg.inv(tg_sigma)),
                           (sur_star - tg_0)[:, np.newaxis]).squeeze()
        ratio_3 = 1/ 2 *(prior_ratio - prior_ratio_star)
        log_ratio = np.sum(ratio_1 + ratio_2) + ratio_3
        ratio = np.exp(log_ratio) if log_ratio < 0 else 1
        # print(ratio)
        rand_ratio = nrd.rand(1)
        if ratio > rand_ratio:
            out = sur_star
            accept_sur += 1
        else:
            out = sur
        return out, accept_sur



    # def update_Sigma(self, w, C):
    #     #Args: Return covariance of random effect w
    #     pm = 3
    #     cov = np.zeros([2, self.NW, self.NW])
    #     for c in range(2):
    #         s = np.where(C == c)
    #         ws = w[s]
    #         pcov = np.identity(self.NW) + np.sum(ws[:, :, np.newaxis] * ws[:, np.newaxis], 0)
    #         cov[c] = ss.invwishart.rvs(pm + s[0].shape[0], pcov)
    #     return cov

    def update_Sigma(self, w, C):
        #Args: Return covariance of random effect w
        pm = 3
        cov = np.zeros([2])
        for c in range(2):
            s = np.where(C == c)
            ws = w[s]
            beta_s = 4 + np.sum(np.square(ws)) / 2    # NW
            alpha_s = ws.shape[0] / 2 + 7  # Prior II
            sigma = 1 / np.random.gamma(alpha_s, 1 / beta_s)
            cov[c] = sigma
        return cov

    # def update_Sigma(self, w):
    #     #Args: Return covariance of random effect w
    #     beta_s = 4 + np.sum(np.square(w)) / 2    # NW
    #     alpha_s = w.shape[0] / 2 + 7  # Prior II
    #     sigma = 1 / np.random.gamma(alpha_s, 1 / beta_s)
    #     return sigma




    def update_tc(self, G, w, m, bs, OT, delta, theta, c_theta, accept_theta):
        ##==============Args: w : [1, intervention, baseline covariates] update theta ==========================================##
        loc_d = np.where(G == 0)
        w = w[loc_d[0]]
        m = m[loc_d[0]]
        OT = OT[loc_d[0]]
        delta = delta[loc_d[0]]
        bs_bl = bs[:self.NB]
        # bs_m = bs[self.NB:]
        fm_bl = self.f_mean(bs_bl, OT)
        # fm_bm = self.f_mean(bs_m, OT)
        int = self.surv_int(bs, fm_bl, center=False)[loc_d[0]]  # N (integral of lambda: survival probability)
        # int = self.surv_tint(OT, bs, center=True)[loc_d[0]]
        gamma = np.sum(w * theta, 1)  # for fixed effect
        egamma = np.exp(gamma)
        ww = np.matmul(w[:, :, np.newaxis], w[:, np.newaxis, :])  # N * NW * NW
        P = np.sum(ww * int[:, np.newaxis, np.newaxis], 0)                                                       # precision matrix
        # =====================if proposal is the same as before, not IWTS ======================================
        theta_star = nrd.multivariate_normal(theta, c_theta * np.linalg.inv(P))
        gamma_star = np.sum(w * theta_star, 1)
        egamma_star = np.exp(gamma_star)
        # prior
        p_theta = np.zeros_like(theta)
        Sigma = np.eye(theta.shape[0])
        # old likelihood and prior
        pitheta = np.sum(gamma * delta - egamma * int)
        p = multivariate_normal.logpdf(theta, mean=p_theta, cov=Sigma)
        # new likelihood and prior
        pitheta_star = np.sum(gamma_star * delta - egamma_star * int)
        p_star = multivariate_normal.logpdf(theta_star, mean=p_theta, cov=Sigma)
        a = pitheta_star + p_star - pitheta - p
        ap = 1 if a >0 else np.exp(a)
        if ap > nrd.rand(1):
            return theta_star, accept_theta+1
        else:
            return theta, accept_theta


    def update_tv(self, w, m, bs, OT, delta,tau, theta, c_bs, accept_m):  # for mediator
        ##-------------------------Agrs: bs: coefficient of spline--------------------------------------##
        D = self.banded([1, -2, 1], self.NB - 2)
        K = np.matmul(np.transpose(D), D)
        # int_bl, int = self.int_bl(OT, bs, m, needed=True)
        fixed = np.exp(np.sum(theta * w, 1))  # N
        bs_bl = bs[:self.NB]
        bs_m = bs[self.NB:]
        fm_bl = self.f_mean(bs_bl, OT)
        fm_bm = self.f_mean(bs_m, OT)
        ##-------------------------calculate Fisher Information ---------------------------------------#
        ##---------------Actual --------------------------------------
        # Sigma = np.zeros(shape=[OT.shape[0], self.NB, self.NB])
        # B = np.zeros(shape=[self.N, self.NTG, self.NB, self.NB])  # for b * b ^t
        # B_bl = np.zeros(shape=[self.N, self.NTG])
        sp_t = BSpline(self.knots, bs_bl, self.K)  # for baseline
        # int_f = np.zeros(shape=[self.N, self.NB, self.NB])
        # for i in range(OT.shape[0]):
        #     b = np.zeros(shape=[self.NTG, self.NB])
        #     timegrid, width = np.linspace(0, OT[i], self.NTG, retstep=True)
        #     for j in range(self.NB):
        #         c = np.zeros(shape=self.NB)
        #         c[j] = 1
        #         sp = BSpline(self.knots, c, self.K)
        #         b[:, j] = sp(timegrid)  # the j-th element of Bspline valued at  timegrid (dim  = NTG)
        #     B = np.matmul(b[:, :, np.newaxis], b[:, np.newaxis, :])  # NTG * NB * NB
        #     B_bl = sp_t(timegrid) - fm_bl   # dim: NTG  bspline function (valued at the timegrid and centered)
        #     f = np.exp(B_bl)[:, np.newaxis, np.newaxis] * B    # NTG * NB * NB
        #     int_f[i] = width * (0.5 * (f[0] + f[-1]) + np.sum(f[1:-1], 0))
        # sigma = int_f * (np.power(m, 2) * fixed)[:, np.newaxis, np.newaxis]
        # Sigma = np.sum(sigma, 0)
        #==================================optimize the Fisher information===========================================#
        # timegrid = np.zeros(shape=[self.N, self.NTG])
        # width = np.zeros(shape=self.N)
        b = np.zeros(shape=[self.N, self.NTG, self.NB])
        # for i in range(OT.shape[0]):
        #     timegrid[i], width[i] = np.linspace(0, OT[i], self.NTG, retstep=True)
        for j in range(self.NB):
            c = np.zeros(shape=self.NB)
            c[j] = 1
            sp = BSpline(self.knots, c, self.K)
            b[:, :, j] = sp(self.timegrid)  # the j-th element of Bspline valued at  timegrid (dim  = NTG)
        B = np.matmul(b[:, :, :, np.newaxis], b[:, :, np.newaxis, :])  # N * NTG * NB * NB
        B_m = sp_t(self.timegrid)- fm_bl   # N * NTG
        f = np.exp(B_m)[:, :, np.newaxis, np.newaxis] * B    # N * NTG * NB * NB
        int_f = self.width[:, np.newaxis, np.newaxis] * (0.5 * (f[:, 0] + f[:, -1]) + np.sum(f[:, 1:-1], 1)) # N * (N * NB * NB) = N* NB * NB
        sigma = int_f * (np.power(m, 2) * fixed)[:, np.newaxis, np.newaxis]
        Sigma = np.sum(sigma, 0) + K / tau[1]
        bs_mstar = nrd.multivariate_normal(bs_m, c_bs[1]*np.linalg.inv(Sigma))
        bs_star = bs.copy()
        bs_star[self.NB:] = bs_mstar
        #--------------------old likelihood, prior and proposal-----------------------
        sp = BSpline(self.knots, bs_m, self.K)
        pibs = (sp(OT) - fm_bm) * m * delta - self.surv_int(bs, m, fm_bl, fm_bm, center=True) * fixed       # N * N = N #likelihood
        # p = multivariate_normal.logpdf(bs_m, mean = np.zeros_like(bs_m), cov= np.power(tau[1],2) * K_inv)  # prior
        p = - 0.5 / tau[1] * np.matmul(np.matmul(bs_m[np.newaxis, :], K), bs_m[:, np.newaxis])
        #=====================new likelihood ==================
        sp_star = BSpline(self.knots, bs_mstar, self.K)
        fm_bmstar = self.f_mean(bs_mstar, OT)
        # pibs_star = (sp_star(OT) - fm_bmstar) * m * delta - self.surv_int(bs_star, m, fm_bl, fm_bmstar, center=True)* fixed
        pibs_star = (sp_star(OT) - fm_bmstar) * m * delta - self.surv_int(bs_star, m, fm_bl, fm_bmstar, center=False) * fixed
        # p_star = multivariate_normal.logpdf(bs_mstar, mean = np.zeros_like(bs_mstar), cov=np.power(tau[1],2) * K_inv)
        p_star = - 0.5 / tau[1] * np.matmul(np.matmul(bs_mstar[np.newaxis, :], K),
                                                           bs_mstar[:, np.newaxis])
        a = np.sum(pibs_star) + p_star - np.sum(pibs) - p
        ap = 1 if a>0 else np.exp(a)
        if ap > nrd.rand(1):
            return bs_star, accept_m+1
        else:
            return bs, accept_m




    def update_alpha(self, m, w, sigma):
            # ---------------- update coefficient --------------------
        w_w = np.sum(np.matmul(w[:, :, np.newaxis], w[:,  np.newaxis, :]), axis=0)
        m_w = np.sum(np.transpose(w) * m, axis=1)
        sigma_alpha = np.linalg.inv(w_w / sigma  + np.diag(v=np.ones(shape=w_w.shape[0]))/100) # 100 is prior
        miu_alpha = np.squeeze(np.dot(sigma_alpha, m_w[:, np.newaxis]) / sigma, 1)
        alpha_star = nrd.multivariate_normal(miu_alpha, sigma_alpha)
        alpha = alpha_star
        return alpha


    def update_G(self, w, theta, beta, bs, OT, delta):
        bs_bl = bs[:self.NB]# 1 means cure group
        beta_w = np.exp(np.sum(beta * w, 1))
        p = beta_w / (1 + beta_w)  # prob of belonging to cure group (G=1)
        fm_bl = self.f_mean(bs_bl, OT)   # here need to be fixed, all mean or the not cure group mean?
        # surv_prob = self.sur_prob(bs, theta, w, OT, fm_bl, center=True)
        tsurv_prob = self.sur_prob(bs, theta, w, OT, fm_bl, center=False)
        a = (1-p) * tsurv_prob
        gp = p / (p + (1-p) * tsurv_prob)
        G = nrd.binomial(1, gp)
        loc_n = np.where(delta == 0)
        loc = np.where(delta == 1)  # have occured, not cure group
        G[loc[0]] = 0
        G_loc = G[loc_n[0]]
        # t_Gloc = t_G[loc_n[0]]
        # aa = np.where(G_loc != t_Gloc)
        # loc_aa = loc_n[0][aa]
        return G

    #============================ here is to update the paramters related to group==================================
    def update_beta(self, G, w, m, theta, c_theta, accept_theta):
        # theta_out = np.zeros(shape=theta.shape[0])
        # sigma_theta = np.zeros(shape=[c_s.shape[2], c_s.shape[2]])
        #=====================================calculate the fisher information===============================
        w_w = np.matmul(np.transpose(w), w)
        pri_Sigma = np.eye(theta.shape[0])
        Sigma = w_w / 4 + np.linalg.inv(pri_Sigma)
        theta_star = nrd.multivariate_normal(theta, c_theta*np.linalg.inv(Sigma))
        #=======================================calculate the ratio=============================================
        loc_d = np.where(G == 0)  # disease group
        loc_c = np.where(G == 1)  # cure group
        theta_w = np.exp(np.sum(theta * w, 1))  # N * 1
        p_c = np.log(theta_w/(1+theta_w))[loc_c[0]]
        p_d = np.log(1/(1+theta_w))[loc_d[0]]
        #====================================prior=================================
        pri_mean = np.zeros_like(theta)
        p = np.sum(p_c) + np.sum(p_d) + multivariate_normal.logpdf(theta, pri_mean, pri_Sigma)
        theta_star_w = np.exp(np.sum(theta_star * w, 1))  # N * 1
        p_sc = np.log(theta_star_w/(1+theta_star_w))[loc_c[0]]
        p_sd = np.log(1/(1+theta_star_w))[loc_d[0]]
        p_s = np.sum(p_sc) + np.sum(p_sd) + multivariate_normal.logpdf(theta_star, pri_mean, pri_Sigma)
        a = p_s - p
        ap = np.exp(a) if a<0 else 1
        if ap > nrd.rand(1):
            return theta_star, accept_theta+1
        else:
            return theta, accept_theta



    def update_delta_alpha(self, x, m, G, delta_alpha, c_delta_alpha, accept_delta_alpha, t_p):
        x_m = np.concatenate((x, m), 1)
        # x_m = x
        for g in range(1, self.NG):
            delta_alpha_star = delta_alpha.copy()
            pre = delta_alpha[g]
            mean = np.matmul(x_m, np.transpose(delta_alpha))
            p = np.exp(mean) / (np.sum(np.exp(mean), 1)[:, np.newaxis])
            pre_star = nrd.multivariate_normal(pre, c_delta_alpha[g] * np.eye(pre.shape[0]))
            # pre_star = pre - 2 / 100
            delta_alpha_star[g] = pre_star
            mean_star = np.matmul(x_m, np.transpose(delta_alpha_star))
            p_star = np.exp(mean_star) / (np.sum(np.exp(mean_star), 1)[:, np.newaxis])
            # loc_0 = np.where(G == 0)
            # loc_1 = np.where(G == 1)
            # loc_2 = np.where(G == 2)
            prior_mean = 0
            prior_var = np.eye(pre.shape[0])
            # ratio_star = np.sum(np.log(p_star[loc_0, 0])) + np.sum(np.log(p_star[loc_1, 1])) + np.sum(np.log(p_star[loc_2, 2])) \
            #              - 1/2 * np.matmul(np.matmul(pre_star - prior_mean, np.linalg.inv(prior_var)), np.transpose(pre_star - prior_mean))
            ratio_star = np.sum(np.log(p_star[np.arange(1000),G.astype(int)])) - \
                         1/2 * np.matmul(np.matmul(pre_star - prior_mean, np.linalg.inv(prior_var)), np.transpose(pre_star - prior_mean))
            # ratio_pre = np.sum(np.log(p[loc_0, 0])) + np.sum(np.log(p[loc_1, 1])) + np.sum(np.log(p[loc_2, 2])) - \
            #             1 / 2 * np.matmul(np.matmul(pre - prior_mean, np.linalg.inv(prior_var)), np.transpose(pre - prior_mean))
            ratio_pre = np.sum(np.log(p[np.arange(1000),G.astype(int)])) - \
                        1 / 2 * np.matmul(np.matmul(pre - prior_mean, np.linalg.inv(prior_var)), np.transpose(pre - prior_mean))
            log_ratio = ratio_star - ratio_pre
            ratio = 1 if log_ratio > 0 else np.exp(log_ratio)
            rand_ratio = nrd.uniform(0, 1)
            if ratio > rand_ratio:
                delta_alpha[g] = pre_star
                accept_delta_alpha[g] += 1
        return delta_alpha, accept_delta_alpha

    #
    # def mediation(self, w, x_a, x_b, x_c, tt, theta, beta, alpha, sigma, bs, t_star):
    #     #--------------------Args:theta PH model; beta: binary model; alpha, sigma: normal model for mediator----------#
    #     #--------------------Args: w 0: intercept; 1: treatment, 2,3: baseline covariates;4:mediator;------------------#
    #     w_b = w[:, :-1]
    #     w_b[:, 1] = x_b
    #     m_mean = np.sum(alpha[:, np.newaxis] * w_b, 2)  # Iter * 1 * NA *( N *NA) =  burnin * N
    #     # simu_m = nrd.normal(np.tile(m_mean[:, :, np.newaxis],(1, 1, self.MC)), sigma[np.newaxis, np.newaxis, np.newaxis]) # Iter * N * MC
    #     simu_m = nrd.normal(np.tile(m_mean[:, :, np.newaxis], (1, 1, self.MC)),  sigma[:, np.newaxis, np.newaxis])  # Iter * N * MC
    #     bs_bl = bs[:self.NB]
    #     hazard = np.zeros(shape=[tt.shape[0], bs_bl.shape[0]])
    #     c_hazard = np.zeros(shape=[tt.shape[0], bs_bl.shape[0]])
    #     for j in range(bs_bl.shape[0]):
    #         sp = BSpline(self.knots, bs_bl[j], self.K)
    #         hazard[:, j] = np.exp(sp(tt))  ##tt.shape * 1 * 1 * 1
    #     # t0 = time.clock()
    #         c_hazard[:, j] = [self.c_hazard(bs_bl[j], i) for i in tt]# cumulative hazard is a function of time tt   NT
    #     # c_hazard = np.array(c_hazard)[:, np.newaxis, np.newaxis, np.newaxis]
    #     # print("One cumulative time: %.9f"%(time.clock() - t0))
    #     hazard = hazard[:, :, np.newaxis, np.newaxis]
    #     c_hazard = c_hazard[:, :, np.newaxis, np.newaxis]
    #     theta_w = np.sum(theta[:, np.newaxis, 1:-1] * w[:, 2:-1], 2) + (x_a * theta[:, 0])[:, np.newaxis]  # (theta * x_a + theta * baseline)  N * 1  Iter * N + Iter * 1 = Iter * N
    #     theta_m = theta[:, -1][:, np.newaxis, np.newaxis] * simu_m      #  (Iter * 1 * 1) * (N * MC) =  Iter * N * MC
    #     tc = theta_w[:, :, np.newaxis] + theta_m      # time-constant part in hazard function(Iter *N *MC)
    #     beta_w = (beta[:, 0] + beta[:, 1] * x_c)[:, np.newaxis] + np.sum(beta[:, np.newaxis, 2:-1] * w[:, 2:-1], 2)   # Iter * 1 + Iter * N = Iter * N
    #     beta_m = beta[:, -1][:, np.newaxis, np.newaxis] * simu_m  # Iter * N * MC
    #     binary = beta_w[:, :, np.newaxis] + beta_m           #Iter * N * MC
    #     p = 1 / (1 + np.exp(binary))[np.newaxis]     # 1 * Iter * N * MC
    #     exp_tc = np.exp(tc)[np.newaxis]
    #     density = hazard * exp_tc * np.exp(-c_hazard * exp_tc) * p   # NT * Iter * N * MC
    #     mdensity = np.mean(density, 3)
    #     survival = (1 - np.exp(-c_hazard * exp_tc)) * p # NT * Iter * N * MC
    #     msurvival = 1 - np.mean(survival, 3)
    #     mhazard = mdensity / msurvival
    #     tstar_grid, star_width = np.linspace(0, t_star, self.NTG, retstep=True)
    #     c_hazard_star = np.zeros(shape=[self.NTG, bs_bl.shape[0]])
    #     for j in range(bs_bl.shape[0]):
    #         c_hazard_star[:, j] =[self.c_hazard(bs_bl[j], i) for i in tstar_grid]# cumulative hazard is a function of time tt   NT
    #         # c_hazard_star = np.array(c_hazard_star)[:, np.newaxis, np.newaxis, np.newaxis]
    #     c_hazard_star = c_hazard_star[:, :, np.newaxis, np.newaxis]
    #     # c_hazard_star = sp.integrate(0, self.tstar_grid)[:, np.newaxis, np.newaxis, np.newaxis] # NTSTAR * 1
    #     int_survival = np.exp(-c_hazard_star * exp_tc)
    #     RMST = star_width * (0.5 * int_survival[0] + 0.5 * int_survival[-1] + np.sum(int_survival[1:-1], 0))  # Iter * N * MC
    #     RMST_b = (RMST - t_star) * np.squeeze(p, 0)
    #     mRMST = np.mean(RMST_b, 2) + t_star
    #     return mdensity, msurvival, mhazard, mRMST     #  NT * Iter * N
    #     # return np.squeeze(mdensity,1), np.squeeze(msurvival,1), np.squeeze(mhazard,1), np.squeeze(mRMST, 0)      # NT * N




    def m_mediation(self, w, x_a, x_b, x_c, tt, theta, beta, alpha, sigma, bs, t_star):
        #--------------------Args:theta PH model; beta: binary model; alpha, sigma: normal model for mediator----------#
        #--------------------Args: w 0: intercept; 1: treatment, 2,3: baseline covariates;4:mediator;------------------#
        w_b = w[:, :-1]
        w_b[:, 1] = x_b
        m_mean = np.sum(alpha[:, np.newaxis] * w_b, 2)  # Iter * 1 * NA *( N *NA) =  burnin * N
        # simu_m = nrd.normal(np.tile(m_mean[:, :, np.newaxis],(1, 1, self.MC)), sigma[np.newaxis, np.newaxis, np.newaxis]) # Iter * N * MC
        simu_m = nrd.normal(np.tile(m_mean[:, :, np.newaxis], (1, 1, self.MC)),  sigma[:, np.newaxis, np.newaxis])  # Iter * N * MC
        bs_bl = bs[:self.NB]
        hazard = np.zeros(shape=[tt.shape[0], bs_bl.shape[0]])
        c_hazard = np.zeros(shape=[tt.shape[0], bs_bl.shape[0]])
        for j in range(bs_bl.shape[0]):
            sp = BSpline(self.knots, bs_bl[j], self.K)
            hazard[:, j] = np.exp(sp(tt))  ##tt.shape * 1 * 1 * 1
        # t0 = time.clock()
            c_hazard[:, j] = [self.c_hazard(bs_bl[j], i) for i in tt]# cumulative hazard is a function of time tt   NT
        # c_hazard = np.array(c_hazard)[:, np.newaxis, np.newaxis, np.newaxis]
        # print("One cumulative time: %.9f"%(time.clock() - t0))
        hazard = hazard[:, :, np.newaxis, np.newaxis]
        c_hazard = c_hazard[:, :, np.newaxis, np.newaxis]
        theta_w = np.sum(theta[:, np.newaxis, 1:-1] * w[:, 2:-1], 2) + (x_a * theta[:, 0])[:, np.newaxis]  # (theta * x_a + theta * baseline)  N * 1  Iter * N + Iter * 1 = Iter * N
        theta_m = theta[:, -1][:, np.newaxis, np.newaxis] * simu_m      #  (Iter * 1 * 1) * (N * MC) =  Iter * N * MC
        tc = theta_w[:, :, np.newaxis] + theta_m      # time-constant part in hazard function(Iter *N *MC)
        beta_w = (beta[:, 0] + beta[:, 1] * x_c)[:, np.newaxis] + np.sum(beta[:, np.newaxis, 2:-1] * w[:, 2:-1], 2)   # Iter * 1 + Iter * N = Iter * N
        beta_m = beta[:, -1][:, np.newaxis, np.newaxis] * simu_m  # Iter * N * MC
        binary = beta_w[:, :, np.newaxis] + beta_m           #Iter * N * MC
        p = 1 / (1 + np.exp(binary))[np.newaxis]     # 1 * Iter * N * MC
        exp_tc = np.exp(tc)[np.newaxis]
        density = hazard * exp_tc * np.exp(-c_hazard * exp_tc) * p   # NT * Iter * N * MC
        mdensity = np.mean(density, (2, 3))  # NT * Iter
        survival = (1 - np.exp(-c_hazard * exp_tc)) * p # NT * Iter * N * MC
        msurvival = 1 - np.mean(survival, (2, 3))  # NT * Iteration
        mhazard = mdensity / msurvival
        tstar_grid, star_width = np.linspace(0, t_star, self.NTG, retstep=True)
        c_hazard_star = np.zeros(shape=[self.NTG, bs_bl.shape[0]])
        for j in range(bs_bl.shape[0]):
            c_hazard_star[:, j] =[self.c_hazard(bs_bl[j], i) for i in tstar_grid]# cumulative hazard is a function of time tt   NT
            # c_hazard_star = np.array(c_hazard_star)[:, np.newaxis, np.newaxis, np.newaxis]
        c_hazard_star = c_hazard_star[:, :, np.newaxis, np.newaxis]
        # c_hazard_star = sp.integrate(0, self.tstar_grid)[:, np.newaxis, np.newaxis, np.newaxis] # NTSTAR * 1
        int_survival = np.exp(-c_hazard_star * exp_tc)
        RMST = star_width * (0.5 * int_survival[0] + 0.5 * int_survival[-1] + np.sum(int_survival[1:-1], 0))  # Iter * N * MC
        RMST_b = (RMST - t_star) * np.squeeze(p, 0)
        mRMST = np.mean(RMST_b, (1, 2)) + t_star  # Iter
        return mdensity, msurvival, mhazard, mRMST     #  NT * Iter
        # return np.squeeze(mdensity,1), np.squeeze(msurvival,1), np.squeeze(mhazard,1), np.squeeze(mRMST, 0)      # NT * N

    def t_mediation(self, w, x_a, x_b, x_c, tt, t_theta, t_beta, t_alpha, t_sigma, t_star):
        #--------------------Args:theta PH model; beta: binary model; alpha, sigma: normal model for mediator----------#
        #--------------------Args: w 0: intercept; 1: treatment, 2,3: baseline covariates;4:mediator;------------------#
        w_b = w[:, :-1]
        w_b[:, 1] = x_b
        m_mean = np.sum(t_alpha * w_b, 1)  #  N
        simu_m = nrd.normal(np.tile(m_mean[:, np.newaxis], (1, self.MC)), t_sigma) # N * MC
        # bs_bl = bs[:self.NB]
        # sp = BSpline(self.knots, bs_bl, self.K)
        # hazard = np.exp(sp(tt))[:, np.newaxis, np.newaxis, np.newaxis]
        # ---------------------------true hazard------------
        hazard = tt[:,np.newaxis, np.newaxis]                # define your own true hazard and cumulative hazard
        c_hazard = (np.power(tt, 2) / 2)[:, np.newaxis, np.newaxis]
        # print("One cumulative time: %.9f"%(time.clock() - t0))
        theta_w = np.sum(t_theta[1:-1] * w[:, 2:-1], 1) + (x_a * t_theta[0])  #  N
        theta_m = t_theta[-1] * simu_m      #  (Iter * 1 * 1) * (N * MC) =  N * MC
        tc = theta_w[:, np.newaxis] + theta_m      # time-constant part in hazard function(N *MC)
        beta_w = (t_beta[0] + t_beta[1] * x_c) + np.sum(t_beta[2:-1] * w[:, 2:-1], 1)   #  N
        beta_m = t_beta[-1] * simu_m  #  N * MC
        binary = beta_w[:, np.newaxis] + beta_m           # N * MC
        p = 1 / (1 + np.exp(binary))   # N * MC
        exp_tc = np.exp(tc)[np.newaxis]
        density = hazard * exp_tc * np.exp(-c_hazard * exp_tc) * p   # NT * N * MC
        mdensity = np.mean(density, 2)  # NT * N
        survival = (1 - np.exp(-c_hazard * exp_tc)) * p   # NT * N * MC
        msurvival = 1 - np.mean(survival, 2)
        mhazard = mdensity / msurvival
        tstar_grid, star_width = np.linspace(0, t_star, self.NTG, retstep=True)
        # c_hazard_star = [self.c_hazard(bs_bl, i) for i in tstar_grid]# cumulative hazard is a function of time tt   NT
        # c_hazard_star = np.array(c_hazard_star)[:, np.newaxis, np.newaxis, np.newaxis]
        # --------------------define your own true cumulative hazard -------------------------------------------------
        c_hazard_star = (np.power(tstar_grid, 2)/2)[:, np.newaxis, np.newaxis]
        int_survival = np.exp(-c_hazard_star * exp_tc)
        RMST = star_width * (0.5 * int_survival[0] + 0.5 * int_survival[-1] + np.sum(int_survival[1:-1], 0))  # N * MC
        RMST_b = (RMST - t_star) * p
        mRMST = np.mean(RMST_b, 1) + t_star
        return mdensity, msurvival, mhazard, mRMST     # NT * N or N(mRMST)

    def tm_mediation(self, w, x_a, x_b, x_c, tt, t_theta, t_beta, t_alpha, t_sigma, t_star):
        # --------------------Args:theta PH model; beta: binary model; alpha, sigma: normal model for mediator----------#
        # --------------------Args: w 0: intercept; 1: treatment, 2,3: baseline covariates;4:mediator;------------------#
        w_b = w[:, :-1]
        w_b[:, 1] = x_b
        m_mean = np.sum(t_alpha * w_b, 1)  # N
        simu_m = nrd.normal(np.tile(m_mean[:, np.newaxis], (1, self.MC)), t_sigma)  # N * MC
        # bs_bl = bs[:self.NB]
        # sp = BSpline(self.knots, bs_bl, self.K)
        # hazard = np.exp(sp(tt))[:, np.newaxis, np.newaxis, np.newaxis]
        # ---------------------------true hazard------------
        hazard = tt[:, np.newaxis, np.newaxis]  # define your own true hazard and cumulative hazard
        c_hazard = (np.power(tt, 2) / 2)[:, np.newaxis, np.newaxis]
        # print("One cumulative time: %.9f"%(time.clock() - t0))
        theta_w = np.sum(t_theta[1:-1] * w[:, 2:-1], 1) + (x_a * t_theta[0])  # N
        theta_m = t_theta[-1] * simu_m  # (Iter * 1 * 1) * (N * MC) =  N * MC
        tc = theta_w[:, np.newaxis] + theta_m  # time-constant part in hazard function(N *MC)
        beta_w = (t_beta[0] + t_beta[1] * x_c) + np.sum(t_beta[2:-1] * w[:, 2:-1], 1)  # N
        beta_m = t_beta[-1] * simu_m  # N * MC
        binary = beta_w[:, np.newaxis] + beta_m  # N * MC
        p = 1 / (1 + np.exp(binary))  # N * MC
        exp_tc = np.exp(tc)[np.newaxis]
        density = hazard * exp_tc * np.exp(-c_hazard * exp_tc) * p  # NT * N * MC
        mdensity = np.mean(density, (1, 2))  # NT
        survival = (1 - np.exp(-c_hazard * exp_tc)) * p  # NT * N * MC
        msurvival = 1 - np.mean(survival, (1, 2))
        mhazard = mdensity / msurvival
        tstar_grid, star_width = np.linspace(0, t_star, self.NTG, retstep=True)
        # c_hazard_star = [self.c_hazard(bs_bl, i) for i in tstar_grid]# cumulative hazard is a function of time tt   NT
        # c_hazard_star = np.array(c_hazard_star)[:, np.newaxis, np.newaxis, np.newaxis]
        # --------------------define your own true cumulative hazard -------------------------------------------------
        c_hazard_star = (np.power(tstar_grid, 2) / 2)[:, np.newaxis, np.newaxis]
        int_survival = np.exp(-c_hazard_star * exp_tc)
        RMST = star_width * (0.5 * int_survival[0] + 0.5 * int_survival[-1] + np.sum(int_survival[1:-1], 0))  # N * MC
        RMST_b = (RMST - t_star) * p
        mRMST = np.mean(RMST_b, (0, 1)) + t_star  # N
        return mdensity, msurvival, mhazard, mRMST  # NT * N or N(mRMST)


    def rm_mediation(self, w, OT, x_a, x_b, x_c, tt, theta, beta, alpha, sigma, bs, t_star):
        #--------------------Args:theta PH model; beta: binary model; alpha, sigma: normal model for mediator----------#
        #--------------------Args: w 0: intercept; 1: treatment, 2,3: baseline covariates;4:mediator;------------------#
        w_b = w[:, :-1]
        w_b[:, 1] = x_b
        m_mean = np.sum(alpha[:, np.newaxis] * w_b, 2)  # Iter * 1 * NA *( N *NA) =  burnin * N
        # simu_m = nrd.normal(np.tile(m_mean[:, :, np.newaxis],(1, 1, self.MC)), sigma[np.newaxis, np.newaxis, np.newaxis]) # Iter * N * MC
        simu_m = nrd.normal(np.tile(m_mean[:, :, np.newaxis], (1, 1, self.MC)),  sigma[:, np.newaxis, np.newaxis])  # Iter * N * MC
        bs_bl = bs[:self.NB]
        # hazard = np.zeros(shape=[tt.shape[0], bs_bl.shape[0]])
        # c_hazard = np.zeros(shape=[tt.shape[0], bs_bl.shape[0]])
        # for j in range(bs_bl.shape[0]):
        #     sp = BSpline(self.knots, bs_bl[j], self.K)
        #     hazard[:, j] = np.exp(sp(tt))  ##tt.shape * 1 * 1 * 1
        #     c_hazard[:, j] = [self.c_hazard(bs_bl[j], i) for i in tt]# cumulative hazard is a function of time tt   NT
        # c_hazard = np.array(c_hazard)[:, np.newaxis, np.newaxis, np.newaxis]
        # # print("One cumulative time: %.9f"%(time.clock() - t0))
        # hazard = hazard[:, :, np.newaxis, np.newaxis]
        # c_hazard = c_hazard[:, :, np.newaxis, np.newaxis]
        theta_w = np.sum(theta[:, np.newaxis, 1:-1] * w[:, 2:-1], 2) + (x_a * theta[:, 0])[:, np.newaxis]  # (theta * x_a + theta * baseline)  N * 1  Iter * N + Iter * 1 = Iter * N
        theta_m = theta[:, -1][:, np.newaxis, np.newaxis] * simu_m      #  (Iter * 1 * 1) * (N * MC) =  Iter * N * MC
        tc = theta_w[:, :, np.newaxis] + theta_m      # time-constant part in hazard function(Iter *N *MC)
        beta_w = (beta[:, 0] + beta[:, 1] * x_c)[:, np.newaxis] + np.sum(beta[:, np.newaxis, 2:-1] * w[:, 2:-1], 2)   # Iter * 1 + Iter * N = Iter * N
        beta_m = beta[:, -1][:, np.newaxis, np.newaxis] * simu_m  # Iter * N * MC
        binary = beta_w[:, :, np.newaxis] + beta_m           #Iter * N * MC
        p = 1 / (1 + np.exp(binary))[np.newaxis]     # 1 * Iter * N * MC
        exp_tc = np.exp(tc)[np.newaxis]  # 1 * Iter * N* MC
        # density = hazard * exp_tc * np.exp(-c_hazard * exp_tc) * p   # NT * Iter * N * MC
        # mdensity = np.mean(density, (2, 3))  # NT * Iter
        # survival = (1 - np.exp(-c_hazard * exp_tc)) * p # NT * Iter * N * MC
        # msurvival = 1 - np.mean(survival, (2, 3))  # NT * Iteration
        # mhazard = mdensity / msurvival
        min_t = np.minimum(OT, t_star)
        # tstar_grid = np.zeros(shape=[OT.shape[0], self.NTG])
        # ttstar_grid = np.zeros(shape=[OT.shape[0], self.NTG, 50])  # N * NTG * 50
        # star_width = np.zeros(shape=[OT.shape[0]])
        # sstar_width = np.zeros(shape=[OT.shape[0], self.NTG])  # N * NTG
         # n = np.zeros(shape=[self.NTG, bs_bl.shape[0], OT.shape[0]])
        c_hazard_star = np.zeros(shape=[self.NTG, bs_bl.shape[0], OT.shape[0]])  # for each N, calculate survival prob for NTG timepoint (0, t)
        width = np.zeros(shape=[OT.shape[0]])
        for i in range(OT.shape[0]):
            tstar_grid, width[i] = np.linspace(0, min_t[i], self.NTG, retstep=True)
            # ttstar_grid, wwidth[i] = np.linspace(0, tstar_grid, 50, retstep=True)
            for j in range(bs_bl.shape[0]):
                c_hazard_star[:, j, i] = [self.c_hazard(bs_bl[j], i) for i in tstar_grid]## cumulative hazard is a function of time tt   N * NTG
        survival = np.exp(-c_hazard_star[:, :, :, np.newaxis] * exp_tc)  # NT * iTER * N * MC
            # c_hazard_star = np.array(c_hazard_star)[:, np.newaxis, np.newaxis, np.newaxis]
        # c_hazard_star = c_hazard_star[:, :, :, np.newaxis]  # NTG * iter *N * 1
        # c_hazard_star = sp.integrate(0, self.tstar_grid)[:, np.newaxis, np.newaxis, np.newaxis] # NTSTAR * 1
        int_survival = (0.5 * survival[0] + 0.5 * survival[-1] + np.sum(survival[1:-1], 0)) * width[np.newaxis, :, np.newaxis]
        # RMST = star_width[np.newaxis,:, np.newaxis] * (0.5 * int_survival[0] + 0.5 * int_survival[-1] + np.sum(int_survival[1:-1], 0))  # Iter * N * MC
        RMST_b = (int_survival - t_star) * np.squeeze(p, 0)
        mRMST = np.mean(RMST_b, (1, 2)) + t_star  # Iter
        mdensity = 0
        msurvival = 0
        mhazard = 0
        return mdensity, msurvival, mhazard, mRMST

    # c_hazard_star = np.zeros(shape=[self.NTG, bs_bl.shape[0]])
    # for j in range(bs_bl.shape[0]):
    #     c_hazard_star[:, j] = [self.c_hazard(bs_bl[j], i) for i in
    #                            tstar_grid]  # cumulative hazard is a function of time tt   NT
    # c_hazard_star = c_hazard_star[:, :, np.newaxis, np.newaxis]
    # # c_hazard_star = sp.integrate(0, self.tstar_grid)[:, np.newaxis, np.newaxis, np.newaxis] # NTSTAR * 1
    # int_survival = np.exp(-c_hazard_star * exp_tc)

    def trm_mediation(self, w, OT, x_a, x_b, x_c, tt, t_theta, t_beta, t_alpha, t_sigma, t_star):
        # --------------------Args:theta PH model; beta: binary model; alpha, sigma: normal model for mediator----------#
        # --------------------Args: w 0: intercept; 1: treatment, 2,3: baseline covariates;4:mediator;------------------#
        w_b = w[:, :-1]
        w_b[:, 1] = x_b
        m_mean = np.sum(t_alpha * w_b, 1)  # N
        simu_m = nrd.normal(np.tile(m_mean[:, np.newaxis], (1, self.MC)), t_sigma)  # N * MC
        # bs_bl = bs[:self.NB]
        # sp = BSpline(self.knots, bs_bl, self.K)
        # hazard = np.exp(sp(tt))[:, np.newaxis, np.newaxis, np.newaxis]
        # ---------------------------true hazard------------
        hazard = tt[:, np.newaxis, np.newaxis]  # define your own true hazard and cumulative hazard
        c_hazard = (np.power(tt, 2) / 2)[:, np.newaxis, np.newaxis]
        # print("One cumulative time: %.9f"%(time.clock() - t0))
        theta_w = np.sum(t_theta[1:-1] * w[:, 2:-1], 1) + (x_a * t_theta[0])  # N
        theta_m = t_theta[-1] * simu_m  # (Iter * 1 * 1) * (N * MC) =  N * MC
        tc = theta_w[:, np.newaxis] + theta_m  # time-constant part in hazard function(N *MC)
        beta_w = (t_beta[0] + t_beta[1] * x_c) + np.sum(t_beta[2:-1] * w[:, 2:-1], 1)  # N
        beta_m = t_beta[-1] * simu_m  # N * MC
        binary = beta_w[:, np.newaxis] + beta_m  # N * MC
        p = 1 / (1 + np.exp(binary))  # N * MC
        exp_tc = np.exp(tc)[:, np.newaxis]  #N * 1 *MC
        # density = hazard * exp_tc * np.exp(-c_hazard * exp_tc) * p  # NT * N * MC
        # mdensity = np.mean(density, (1, 2))  # NT
        # survival = (1 - np.exp(-c_hazard * exp_tc)) * p  # NT * N * MC
        # msurvival = 1 - np.mean(survival, (1, 2))
        # mhazard = mdensity / msurvival
        min_t = np.minimum(OT, t_star)
        tstar_grid = np.zeros(shape=[OT.shape[0], self.NTG])
        star_width = np.zeros(shape=[OT.shape[0]])
        for i in range(OT.shape[0]):
            tstar_grid[i], star_width[i] = np.linspace(0, min_t[i], self.NTG, retstep=True)
        # c_hazard_star = [self.c_hazard(bs_bl, i) for i in tstar_grid]# cumulative hazard is a function of time tt   NT
        # c_hazard_star = np.array(c_hazard_star)[:, np.newaxis, np.newaxis, np.newaxis]
        # --------------------define your own true cumulative hazard -------------------------------------------------
        c_hazard_star = (np.power(tstar_grid, 2) / 2)[:, :, np.newaxis]   # N * NTG * 1
        int_survival = np.exp(-c_hazard_star * exp_tc)  # N * NTG * MC
        RMST = star_width[:, np.newaxis] * (0.5 * int_survival[:, 0] + 0.5 * int_survival[:, -1] + np.sum(int_survival[:, 1:-1], 1))  # N * MC
        RMST_b = (RMST - t_star) * p
        mRMST = np.mean(RMST_b, (0, 1)) + t_star  # N
        mdensity = 0
        msurvival = 0
        mhazard = 0
        return mdensity, msurvival, mhazard, mRMST  # NT * N or N(mRMST)

    def arm_mediation(self, w, OT, x_a, x_b, x_c, theta, beta, alpha, sigma, bs, t_star):
        # --------------------Args:theta PH model; beta: binary model; alpha, sigma: normal model for mediator----------#
        # --------------------Args: w 0: intercept; 1: treatment, 2,3: baseline covariates;4:mediator;------------------#
        #----------------------Return survival prob and hazard prob at t_star, RMST---------------------------
        w_b = w[:, :-1]
        w_b[:, 1] = x_b
        m_mean = np.sum(alpha[:, np.newaxis] * w_b, 2)  # Iter * 1 * NA *( N *NA) =  burnin * N
        # simu_m = nrd.normal(np.tile(m_mean[:, :, np.newaxis],(1, 1, self.MC)), sigma[np.newaxis, np.newaxis, np.newaxis]) # Iter * N * MC
        simu_m = nrd.normal(np.tile(m_mean[:, :, np.newaxis], (1, 1, self.MC)),
                            sigma[:, np.newaxis, np.newaxis])  # Iter * N * MC
        bs_bl = bs[:, :self.NB]
        # min_t = np.minimum(OT, t_star)
        bl_hazard = np.zeros(shape=[bs_bl.shape[0], OT.shape[0]])  # baseline hazard (Iter * N)
        # width = np.zeros(shape=[OT.shape[0]])
        c_hazard_star = np.zeros(shape=[self.NTG, bs_bl.shape[0]])
        tstar_grid, star_width = np.linspace(0, t_star, self.NTG, retstep=True)
        for j in range(bs_bl.shape[0]):
            sp = BSpline(self.knots, bs_bl[j], self.K)
            bl_hazard[j] = np.exp(sp(t_star))
            c_hazard_star[:, j] = [self.c_hazard(bs_bl[j], i) for i in tstar_grid]  ## cumulative hazard is a function of time tt   N * NTG
        # c_hazard = np.array(c_hazard)[:, np.newaxis, np.newaxis, np.newaxis]
        # # print("One cumulative time: %.9f"%(time.clock() - t0))
        # hazard = hazard[:, :, np.newaxis, np.newaxis]
        # c_hazard = c_hazard[:, :, np.newaxis, np.newaxis]
        theta_w = np.sum(theta[:, np.newaxis, 1:-1] * w[:, 2:-1], 2) + (x_a * theta[:, 0])[:,
                                                                       np.newaxis]  # (theta * x_a + theta * baseline)  N * 1  Iter * N + Iter * 1 = Iter * N
        theta_m = theta[:, -1][:, np.newaxis, np.newaxis] * simu_m  # (Iter * 1 * 1) * (N * MC) =  Iter * N * MC
        tc = theta_w[:, :, np.newaxis] + theta_m  # time-constant part in hazard function(Iter *N *MC)
        exp_tc = np.exp(tc) #    Iter * N* MC
        beta_w = (beta[:, 0] + beta[:, 1] * x_c)[:, np.newaxis] + np.sum(beta[:, np.newaxis, 2:-1] * w[:, 2:-1],
                                                                         2)  # Iter * 1 + Iter * N = Iter * N
        beta_m = beta[:, -1][:, np.newaxis, np.newaxis] * simu_m  # Iter * N * MC
        binary = beta_w[:, :, np.newaxis] + beta_m  # Iter * N * MC
        p = 1 / (1 + np.exp(binary))  # Iter * N * MC   ( belong to non-cure group)
        survival = np.exp(-c_hazard_star[:, :, np.newaxis, np.newaxis] * exp_tc[np.newaxis])  # NT * iTER * N * MC
        hazard = exp_tc * bl_hazard[:, :, np.newaxis] * p  # first belong to this group) # Iter *N *MC
        # m_hazard = np.squeeze(np.mean(hazard, 3), 0)  # Iter * N
        m_hazard = np.mean(hazard, (1, 2)) # Iter
        gsurvival = survival[-1] * p + 1 * (1 - p)  # sur prob for cure is 1 (survival[-1]: survival prob at t_star)  # Iter *N * MC
        m_survial = np.mean(gsurvival, (1, 2))
        int_survival = (0.5 * survival[0] + 0.5 * survival[-1] + np.sum(survival[1:-1], 0)) * star_width
        # RMST = star_width[np.newaxis,:, np.newaxis] * (0.5 * int_survival[0] + 0.5 * int_survival[-1] + np.sum(int_survival[1:-1], 0))  # Iter * N * MC
        RMST_b = (int_survival - t_star) * p
        mRMST = np.mean(RMST_b, (1, 2)) + t_star  # Iter
        return m_survial, m_hazard, mRMST

    def tarm_mediation(self, w, OT, x_a, x_b, x_c, t_theta, t_beta, t_alpha, t_sigma, t_star):
        # --------------------Args:theta PH model; beta: binary model; alpha, sigma: normal model for mediator----------#
        # --------------------Args: w 0: intercept; 1: treatment, 2,3: baseline covariates;4:mediator;------------------#
        w_b = w[:, :-1]
        w_b[:, 1] = x_b
        m_mean = np.sum(t_alpha * w_b, 1)  # N
        simu_m = nrd.normal(np.tile(m_mean[:, np.newaxis], (1, self.MC)), t_sigma)  # N * MC
        # bs_bl = bs[:self.NB]
        # sp = BSpline(self.knots, bs_bl, self.K)
        # hazard = np.exp(sp(tt))[:, np.newaxis, np.newaxis, np.newaxis]
        theta_w = np.sum(t_theta[1:-1] * w[:, 2:-1], 1) + (x_a * t_theta[0])  # N
        theta_m = t_theta[-1] * simu_m  # (Iter * 1 * 1) * (N * MC) =  N * MC
        tc = theta_w[:, np.newaxis] + theta_m  # time-constant part in hazard function(N *MC)
        beta_w = (t_beta[0] + t_beta[1] * x_c) + np.sum(t_beta[2:-1] * w[:, 2:-1], 1)  # N
        beta_m = t_beta[-1] * simu_m  # N * MC
        binary = beta_w[:, np.newaxis] + beta_m  # N * MC
        p = 1 / (1 + np.exp(binary))  # N * MC
        exp_tc = np.exp(tc)  #N  *MC
        # ---------------------------true hazard------------
        # min_t = np.minimum(OT, t_star)  # N
        # bl_hazard = (np.power(t_star, 2) / 2)  # a number
        bl_hazard = t_star + 1
        hazard = bl_hazard * exp_tc * p  # N * MC hazard at t_star)
        m_hazard = np.mean(hazard)
        tstar_grid, star_width = np.linspace(0, t_star, self.NTG, retstep=True)
        # c_hazard_star = [self.c_hazard(bs_bl, i) for i in tstar_grid]# cumulative hazard is a function of time tt   NT
        # c_hazard_star = np.array(c_hazard_star)[:, np.newaxis, np.newaxis, np.newaxis]
        # --------------------define your own true cumulative hazard -------------------------------------------------
        c_hazard_star = (np.power(tstar_grid, 2) / 2 + tstar_grid)[np.newaxis, :, np.newaxis]   # 1 * NTG * 1
        survival = np.exp(-c_hazard_star * exp_tc[:, np.newaxis])  # N * NTG * MC    #### survival at different timepoints
        gsurvival = survival[:, -1] * p + (1 - p)  # N * MC
        m_survival = np.mean(gsurvival)
        RMST = star_width * (0.5 * survival[:, 0] + 0.5 * survival[:, -1] + np.sum(survival[:, 1:-1], 1))  # N * MC
        RMST_b = (RMST - t_star) * p
        mRMST = np.mean(RMST_b, (0, 1)) + t_star  # N
        return m_survival, m_hazard, mRMST  # NT * N or N(mRMST)


    def m_BIC(self, y, x, xi, lam, phi, w, Sigma, bs, C, f, sur, OT, nu, grid, delta, lambd, d, inc):
        # The args is the mean parameter
        ##----------------------------------Factor analysis model-------------------------------
        eta = self.update_eta(bs, w)
        y_likeli = self.y_likeli(y, x, xi, eta, lam, phi)
        ##-----------Split into two groups-------------------
        cure = np.where(C == 1)[0]
        cure_ylikeli = y_likeli[1, cure]
        no_cure = np.where(C == 0)[0]
        nocure_ylikeli = y_likeli[0, no_cure]
        #######----------------------- hazard model-------------------------------------###
        delta_sample = delta[no_cure]
        loc_no = np.where(delta_sample == 0)[0]
        no_sample = no_cure[loc_no]  #no cured and not happened
        loc_h = np.where(delta_sample == 1)[0]
        h_sample = no_cure[loc_h]   # no cured and happend
        n_c = np.concatenate((f, xi[:, 0], w[:, np.newaxis]), 1)
        cond = np.sum(sur * n_c, 1)
        # exp_cond = np.exp(cond)
        hazard = 0  # for happend sample
        h_nu = nu[h_sample]
        happen = np.where(h_nu == 1)
        log_hazard = np.log(lambd[happen[1]])
        sur_prob = self.sur_prob(f, xi, w, sur, OT, lambd, nu, grid)
        happ_likeli = log_hazard + cond[h_sample] + sur_prob[h_sample]
        nohapp_likeli = sur_prob[no_sample]
        #------------------------------Incidence model---------------------------------##
        n_d = np.concatenate((d, xi[:, 0]), axis=1)
        inc_d = np.sum(n_d * inc, 1)
        inc_likeli = C * inc_d - np.log(1 + np.exp(inc_d))
        #--------------------------------Random effect model-----------------------------##
        w_likeli = self.w_likeli(w, Sigma)
        cure_wlikeli = w_likeli[1, cure]
        nocure_wlikeli = w_likeli[0, no_cure]
        L = np.sum(cure_ylikeli) + np.sum(nocure_ylikeli) + np.sum(happ_likeli) + np.sum(nohapp_likeli) +\
            np.sum(inc_likeli) + np.sum(cure_wlikeli) + np.sum(nocure_wlikeli)
        return L


















