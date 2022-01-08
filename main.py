from __future__ import division
from scipy.optimize import fsolve, root
from scipy import integrate
from sympy import Symbol, exp, log
from scipy.interpolate import BSpline
import matplotlib.pyplot as plt
import numpy as np
import tool
import update
import time
import scipy.stats
import numpy.random as nrd
import multiprocessing

N = 1000
NT = 5    # max number of observed time
Iter = 6000
burnin = 3000
NX = 1
NY = 3
# NH = 3
NW = 1    ###### number of random effect for design time matrix
ND = 2
NF = 2
NXI = 3
NETA = 1  # number of  latent variabel
NG = 5  # number of grid
NK = 15  # number of inner knots
K = 2  # order (more explitily, degree (cubic)  (so number of spline is NK+K)
NB = NK + K - 1  # number of b-spline
NTRA = NX + NXI
NINC = ND + 1 + NXI
NSUR = NF + NXI + NW
Rep = 1
TNXI = 3

def do(o):
    name = multiprocessing.current_process().name
    print("name : %s staring, n:%d"% (name, o),  flush=True, file=open("sur.txt", "a"))
    Rep_s = o * 1
    Rep_e = (o + 1) * 1
    Rep = Rep_e - Rep_s
    # -------------------------------------To save parameters----------------------------------
    all_C = np.zeros(shape=[Rep, Iter, N])
    all_lam = np.zeros([Rep, Iter, 2, NY, NX + NXI + NETA+1])
    all_phi = np.zeros([Rep, Iter, 2, NY])
    # all_tra = np.zeros([Rep, Iter, 2, NTRA])
    all_inc = np.zeros([Rep, Iter, NINC])
    all_lambd = np.zeros([Rep, Iter, NG])
    all_sur = np.zeros([Rep, Iter, NF + NXI + NW])
    all_Sigma = np.zeros([Rep, Iter, 2])
    all_w = np.zeros([Rep, Iter, N])
    all_bs = np.zeros([Rep, Iter, 2, NB])
    all_tau = np.zeros([Rep, Iter, 2])
    for rep in range(Rep_s, Rep_e):
        # rep = 2
        nrd.seed(39)
        # #---------------------------------------------------------Segment line---------------------------------------##
        ##################---------------------------------------This is for loading data--------------------------##
        d = np.load("all_d.npy")[rep]
        t_C = np.load("all_tc.npy")[rep]
        all_xi = np.load("all_xi.npy")[rep]
        xi = all_xi[:, :, :NXI]
        t_eta = np.load("all_teta.npy")[rep]
        x = np.load("all_x.npy")[rep]
        y = np.load("all_y.npy")[rep]
        f = np.load("all_f.npy")[rep]
        t_w = np.load("all_tw.npy")[rep]
        OT = np.load("all_OT.npy")[rep]
        CT = np.load("all_CT.npy")[rep]
        ET = np.load("all_ET.npy")[rep]
        t_set = np.load("all_tset.npy")[rep]
        OBS_T = np.load("all_OBST.npy")[rep]
        delta = (CT >= ET) + 0  # observed failure (dead) is 1
        tt = np.arange(NT)
        logic_obs = np.repeat(tt[np.newaxis], N, axis=0) < np.repeat(OBS_T[:, np.newaxis], NT,
                                                                     axis=1)  # if observed obs is True
        O = logic_obs.astype(int)
        # # #---------------------------------------------Set true parameters---------------------------------------
        t_Sigma = np.array([1, 0.64])
        t_theta = np.array([-2, 1, 1])
        t_alpha = np.array([-1, -1, -1, -1])
        t_A = np.zeros([2, NY, TNXI])
        t_A[0] = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        t_A[1] = [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]
        t_mu = np.zeros([2, NY, NX + 1])
        t_mu[0] = np.array([[2, 1], [2, 1], [2, 1]])
        t_mu[1] = np.array([[-2, -1], [-2, -1], [-2, -1]])
        t_lam = np.array([[[1], [0.5], [0.5]],
                          [[1], [0.6], [0.6]]])  # 2* NY * NETA
        t_phi = np.array([[0.25, 0.25, 0.25], [0.36, 0.36, 0.36]])
        t_mu_lam = np.concatenate((t_mu, t_lam), 2)
        t_Lam = np.concatenate((t_A, t_mu, t_lam), 2)  # 2 * NY * (NX + NXI + NETA +1)
        t_phi = np.array([[0.25, 0.25, 0.25], [0.36, 0.36, 0.36]])
        t_psi = np.array([-1, -0.5])  ## coefficient for x, basline1, baseline2, and m
        t_tau = np.array([-0.5, -0.5])
        t_pi = np.array([-0.5])
        # ----------------------Initalize parameters--------------------------------------
        # ---------------------------Factor analysis model--------------------
        lam = nrd.normal(0, 1, (2, NY, NXI + NX + NETA + 1))
        lam[:, 0, NXI + NX + NETA] = 1
        phi = nrd.uniform(0, 1, (2, NY))
        Ind = np.ones(shape=[NY,
                             NXI + NX + NETA + 1])  # to show lambda is fixed or not (dim : NY * 3) fix is zero ( for each state is the same)
        Ind[0, NXI + NX + NETA] = 0  # fixed
        # #---------------------Trajectory model-----------------
        # tra = nrd.normal(0, 1, (2, NTRA))
        # t_tra = np.concatenate((t_beta, t_gamma, t_zeta), axis=1)
        # c_tra = np.array([0.1, 0.1])
        accept_tra = np.zeros(shape=[2])
        # sigma = nrd.uniform(0, 1, 2)
        # eta = nrd.normal(0, 1, (N, NT, NETA))
        # -----------------------Bspline model---------------------
        knots = np.zeros(shape=[NK + 2 * K])
        knots[K:(NK + K)] = np.linspace(0, 10, NK, endpoint=True)
        knots[:K] = 0
        knots[NK + K:] = 10
        bs = nrd.randn(2, NB)
        c_bs = np.array([0.0001, 0.0005])
        accept_bs = np.zeros([2])
        tau = nrd.uniform(0, 1, size=2)
        # #---------------------Logistic incidence model------------
        t_inc = np.concatenate((t_theta, t_alpha))
        inc = nrd.normal(0, 1, NINC)
        c_inc = 1
        accept_inc = 0
        # #-----------------------------Survival model: piecewise-------------------------##
        c_l = np.max(OT) + 1
        grid = np.zeros(shape=[NG + 1])
        grid[-1] = c_l  # t_G > OT all
        for g in range(1, NG):
            grid[g] = np.percentile(OT, g / NG * 100)
        nu = np.zeros(shape=[N, NG])
        tool1 = tool.Tools(N, NT)
        # b_igt = tool1.b_integration(np.min(t_set), np.max(t_set))
        interval = tool1.find_obs_num(OT, grid)
        nu[np.arange(N), interval - 1] = 1
        t_sur = np.concatenate((t_psi, t_tau, t_pi))
        lambd = nrd.uniform(0, 3, NG)
        t_lambd = np.ones([NG])
        sur = nrd.normal(0, 1, NF + NXI + NW)
        c_sur = 1
        accept_sur = 0
        sample = np.where(delta == 0)  # not happened
        C = nrd.binomial(1, p=0.5, size=N)
        event = np.where(delta == 1)
        C[event[0]] = 0
        # #-----------------------------------random effect---------------------
        # c_w = np.array([0.2, 0.1])
        c_w = np.array([1])
        accept_w = np.zeros([N])
        w = nrd.normal(0, 1, N)
        ######-------------------------------------------------MCMC method to solve------------------------------------------##
        data = update.MCMC(N, NT, NX, NY, NW, ND, NF, NTRA, NETA, NG, O, OBS_T, t_set, NK, NB, K, knots)
        # ------------------------------Replication--------------------
        r = rep - Rep_s
        for iter in range(Iter):
            t0 = time.time()
            # # #-------------------Bspline model---------------------------------------
            bs, accept_bs = data.update_bs(C, y, x, xi, w, bs, lam, phi, tau, c_bs, accept_bs)
            all_bs[r, iter] = bs
            tau = data.update_tau(bs)
            all_tau[r, iter] = tau
            # # # #---------------------Factor analysis model------------------------------#
            lam, phi = data.update_lam_phi(C, y, phi, lam, Ind, x, xi, w, bs)
            all_lam[r, iter] = lam
            all_phi[r, iter] = phi
            # # print(phi)
            # ------------------Incidence model----------------------------------
            inc, accept_inc = data.update_incidence(d, xi, w, C, inc, c_inc, accept_inc)
            all_inc[r] = inc
            # print(inc)
            # # #----------------------Survival model-------------------------------------
            lambd = data.update_lambd(lambd, nu, delta, sur, f, xi, w, OT, grid, C)
            all_lambd[r, iter] = lambd
            sur, accept_sur = data.update_sur(sur, f, xi, w, delta, OT, nu, lambd, grid, c_sur, accept_sur, C)
            all_sur[r, iter] = sur
            # print(sur)
            # # #----------------------random effect--------------------------------------------#
            Sigma = data.update_Sigma(w, C)
            all_Sigma[r, iter] = Sigma
            # print(Sigma)
            w, accept_w = data.update_w(C, xi, y, lam, phi, x, f, sur, OT, lambd, nu, grid, delta, w, bs, Sigma, c_w,
                                        accept_w)
            all_w[r, iter] = w
            # # # # # #------------------------Update group label-----------------------------------
            eta = data.update_eta(bs, w)
            y_likeli = data.y_likeli(y, x, xi, eta, lam, phi)
            w_likeli = data.w_likeli(w, Sigma)
            ep = data.pc(d, xi, w, inc)
            sur_prob = data.sur_prob(f, xi, w, sur, OT, lambd, nu, grid)
            C = data.update_C(y_likeli, sur_prob, ep, sample, w_likeli)
            all_C[r, iter] = C
            acc = np.sum(C == t_C) / N
            # aa = np.where(C[sample] != t_C[sample])
            # sa = sample[0][aa]  # label index in all population
            acc_n = np.sum(C[sample] == t_C[sample]) / sample[0].shape[0]
            process = (r * Iter + iter) / (Rep * Iter)
            one_iter_time = time.time() - t0
            if iter > 450 and iter % 500 == 0:
                print("%.3f seconds process time for one iter" % one_iter_time)
                print("%.3f seconds process time for one iter" % one_iter_time, flush=True, file=open("sur.txt", "a"))
                # print("Acceptance for sur, w and tra" % (accept_sur, accept_w, accept_tra), flush=True, file=open("sur.txt", "a"))
                rtime = Rep * Iter * one_iter_time * (1 - process) / 60
                print("Rep :%d, Iter : %d, process: %.3f, need %.1f min to complete" % (r, iter, process, rtime))
                print("acc of cured label for all and for censoring group is %.3f, %.3f" % (acc, acc_n))
                print("acc of cured label for all and for censoring group is %.3f, %.3f" % (acc, acc_n), flush=True,
                      file=open("sur.txt", "a"))
                print("Rep :%d, Iter : %d, process: %.3f, need %.1f min to complete" % (r, iter, process, rtime),
                      flush=True,
                      file=open("sur.txt", "a"))
    print("Replication is over")
    np.save("Sigma" + str(o) + ".npy", all_Sigma)
    np.save("w" + str(o) + ".npy", all_w)
    np.save("sur" + str(o) + ".npy", all_sur)
    np.save("lam" + str(o) + ".npy", all_lam)
    np.save("lambd" + str(o) + ".npy", all_lambd)
    np.save("phi" + str(o) + ".npy", all_phi)
    np.save("inc" + str(o) + ".npy", all_inc)
    np.save("C" + str(o) + ".npy", all_C)
    np.save("bs" + str(o) + ".npy", all_bs)


if __name__ == '__main__':
    numList = []
    for o in range(0, 1):
        p = multiprocessing.Process(target=do, args=(o,))
        numList.append(p)
        p.start()
        p.join()
