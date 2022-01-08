from __future__ import division
from scipy.optimize import fsolve, root
from scipy import integrate
from sympy import Symbol, exp, log
from scipy.interpolate import BSpline
from numpy import linalg as la
import matplotlib.pyplot as plt
import numpy as np
import tool
import update
import time
import scipy.stats
import numpy.random as nrd
import multiprocessing

N = 500
NT = 3    # max number of observed time
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
hh = 30
ww = 30
L = 30
# #-------------------------To save data-------------------------
# all_tw = np.zeros([Rep, N, NY * NW])
# all_xi = np.zeros([Rep, N, NXI])
# all_d = np.zeros([Rep, N, ND + 1])
# all_tb = np.zeros([Rep, N, 1])
# all_ty = np.zeros([Rep, N, NT])
# all_x = np.zeros([Rep, N, NT, NX + 1])
# all_f = np.zeros([Rep, N, NF])
# all_ET = np.zeros(shape=[Rep, N])
# all_CT = np.zeros(shape=[Rep, N])
# all_OT = np.zeros(shape=[Rep, N])
# all_delta = np.zeros(shape=[Rep, N])
# all_tC = np.zeros(shape=[Rep, N])

def do(o):
    name = multiprocessing.current_process().name
    print("name : %s staring, n:%d"% (name, o),  flush=True, file=open("sur.txt", "a"))
    Rep_s = o * 10
    Rep_e = (o + 1) * 10
    Rep = Rep_e - Rep_s
    # -------------------------------------To save parameters----------------------------------
    all_C = np.zeros(shape=[Rep, Iter, N])
    all_lam = np.zeros([Rep, Iter, 2, NY, NX + NXI+NETA+1])
    all_phi = np.zeros([Rep, Iter, 2, NY])
    # all_tra = np.zeros([Rep, Iter, 2, NTRA])
    all_inc = np.zeros([Rep, Iter, NINC])
    all_lambd = np.zeros([Rep, Iter, NG])
    all_sur = np.zeros([Rep, Iter, NF + NXI + NW])
    all_Sigma = np.zeros([Rep, Iter, 2])
    all_w = np.zeros([Rep, Iter, N])
    all_bs = np.zeros([Rep, Iter, 2, NB])
    all_tau = np.zeros([Rep, Iter, 2])
    # all_A = np.zeros([Rep, Iter, 2, NY, NXI])
    # all_vest_eimg = np.zeros([Rep, NXI, hh*ww])
    all_est_ev = np.zeros([Rep, NXI])
    for rep in range(Rep_s, Rep_e):
        r = rep - Rep_s
        #############################---------------------------------FPCA(to be completed)-------------------------------#######
        #####_---------------------------------------Eigenimage----------------------------------##################
        t_eimg = np.zeros(shape=[TNXI, hh, ww])
        t_eimg[0, :10, :10] = 1
        t_eimg[1, 10:20, 10:20] = 1
        t_eimg[2, 20:30, 20:30] = 1
        t_eimg = t_eimg / np.sqrt(np.sum(t_eimg, (1, 2)))[:, np.newaxis, np.newaxis]
        ev = np.array([1, 0.5, 0.25])
        t_xi = np.zeros(shape=[N, NT, TNXI])
        t_xi[:, :, 0] = nrd.normal(0, np.sqrt(ev[0]), (N, NT))
        t_xi[:, :, 1] = nrd.normal(0, np.sqrt(ev[1]), (N, NT))
        t_xi[:, :, 2] = nrd.normal(0, np.sqrt(ev[2]), (N, NT))  # real eigenscore
        OBS_T = nrd.randint(3, 6, N)  # (OT is 3, 4, 5)
        # OT = np.ones(N) * NT
        tt = np.arange(NT)
        logic_obs = np.repeat(tt[np.newaxis], N, axis=0) < np.repeat(OBS_T[:, np.newaxis], NT,
                                                                     axis=1)  # if observed obs is True
        O = logic_obs.astype(int)
        obs = np.where(logic_obs == True)
        no_obs = np.where(logic_obs == False)
        vlogic_obs = logic_obs.reshape(-1)  # vectorized (N*NT)
        vno_obs = np.where(vlogic_obs == False)
        v_obs = np.where(vlogic_obs == True)
        t_xi[no_obs[0], no_obs[1]] = 0
        obs_txi = t_xi[obs[0], obs[1]]
        t_img = np.sum(t_xi[:, :, :, np.newaxis, np.newaxis] * t_eimg, 2)  # This is the image (N * NT * h * w)
        v_img = t_img.reshape((N * NT, hh * ww))  # NUMBER * hw
        obs_num = int(np.sum(OBS_T))
        # -----------------------------------HDFPCA-----------------------------------------------
        obs_vimg = v_img[v_obs]  # obs_num * grid
        obs_vmean = np.mean(obs_vimg, 0)  # obeserved mean of vectorized image
        obs_vimg = obs_vimg - obs_vmean[np.newaxis]  # standarized
        tool1 = tool.Tools(N, NT)
        L = 30
        img_square = tool1.large_square(obs_vimg.transpose(), L)
        u, s, ut = la.svd(img_square)  # u: n*n, s: n (eigenvalue)
        est_ev = s / obs_num  # estimated eigenvalue
        all_est_ev[r] = est_ev[:NXI]
        print("SVD done in %r replication" % rep, flush=True, file=open("sur.txt", "a"))
        obs_xi = np.matmul(np.diag(np.sqrt(s)),
                           ut)  # Estimated eigenscore   ( can truncated to   Trun* n  == eigenscore[:Trun]  Eigenscore)
        obs_xi = obs_xi[:NXI, :].transpose()  # Truncated to nxi
        # axi = np.matmul(np.diag(np.sqrt(s)), np.transpose(u))      xi and axi is the same
        pg = int(hh * ww / L)  # grid number of each part
        est_eimg = np.zeros(shape=[obs_num, L, pg])
        # for l in range(L):
        #     y_part = obs_vimg[:, l * pg:(l + 1) * pg]
        #     est_eimg[:, l] = (np.matmul(y_part.transpose(), np.matmul(u, np.diag(1/np.sqrt(s))))).transpose()  # eigenimage   [N * L * PG)
        # vest_eimg = est_eimg.reshape((obs_num, -1))  # vectorize the eigenimage [:5, :]  # truncated to first nrow dime
        # all_vest_eimg[r] = vest_eimg[:NXI]
        # v_eimg = t_eimg.reshape((3, -1))  # vectorized value is also OK
        # ------------------------Reshape xi to real xi------------------------
        xi = np.zeros([N, NT, NXI])
        v_xi = xi.reshape((N * NT), -1)
        v_xi[v_obs] = obs_xi
        xi = v_xi.reshape((N, NT, -1))
        #######################--------------------------------------------logistic incidence model -------------------------#####-
        # ---------------------------------------------Random effect--------------------------------------
        d = np.zeros(shape=[N, ND + 1])
        d[:, 0] = 1
        d[:, 1] = nrd.binomial(1, p=0.4, size=N)
        d[:, 2] = nrd.normal(0, 1, N)
        t_theta = np.array([-2, 1, 1])
        theta_d = np.sum(t_theta * d, 1)
        t_alpha = np.array([-1, -1, -1])
        # f_alpha = np.sum(t_alpha[:, np.newaxis, np.newaxis] * t_eimg, 0)  # h *w
        # aa = np.sum(f_alpha * t_img[:, 0], (1,2))
        alpha_xi = np.sum(t_alpha * t_xi[:, 0], 1)
        # t_a = np.array([-1])
        m = theta_d + alpha_xi
        p = np.exp(m) / (1 + np.exp(m))
        t_C = nrd.binomial(1, p=p)
        NC = np.sum(t_C == 1)
        NO = np.sum(t_C == 0)
        fraction = NC / N  # cure fraction
        print("percent of cure group is %.3f" % fraction, flush=True, file=open("data.txt", "a"))
        ######-----------------------------------------Trajectory Model--------------------------
        # t_set = np.array([1.5, 2, 3, 5, 6, 8, 9])
        t_set = np.sort(nrd.uniform(0, 10, size=(N, NT)), axis=1)
        t_design = np.zeros(shape=[2, N, NT])  # different for two group
        t_design[0] = np.sqrt(t_set)  # This is f(t) for nocured group
        t_design[1] = np.log(t_set + 1)  # This is f(t) for cured group
        t_design = t_design - np.mean(t_design, (1, 2))[:, np.newaxis, np.newaxis]
        cure = np.where(t_C == 1)
        no_cure = np.where(t_C == 0)
        # t_beta = np.array([[1, 1], [0.5, 0.5]])
        # # --------------------------latent variable Deal by two group-------------------------------
        t_Sigma = np.array([1, 0.64])
        t_w = np.zeros(shape=[N])
        t_w[no_cure] = nrd.normal(0, np.sqrt(t_Sigma[0]), NO)
        t_w[cure] = nrd.normal(0, np.sqrt(t_Sigma[1]), NC)
        # t_eta = np.zeros(shape=[N, NT, NETA])
        m_eta = np.zeros(shape=[N, NT, NETA])
        m_eta[no_cure[0], :, 0] = t_design[0, no_cure[0]] + t_w[no_cure[0], np.newaxis]
        # # ---------------------------For cured group-------------
        m_eta[cure[0], :, 0] = t_design[1, cure[0]] + t_w[cure[0], np.newaxis]
        t_eta = m_eta
        ######--------------------------Factor analysis model--------------------------------------------####
        # t_mu = np.array([[2, 2, 2], [-2, -2, -2]])  # 2 * NY
        t_lam = np.array([[[1], [0.5], [0.5]],
                          [[1], [0.6], [0.6]]])  # 2* NY * NETA
        t_phi = np.array([[0.25, 0.25, 0.25], [0.36, 0.36, 0.36]])
        x = np.zeros([N, NT, NX + 1])
        x[:, :, 0] = 1
        x[:, :, 1] = nrd.normal(0, 1, (N, NT))
        y_mean = np.zeros([N, NT, NY])
        y = np.zeros([N, NT, NY])
        t_A = np.zeros([2, NY, TNXI])
        t_A[0] = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        t_A[1] = [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]
        t_mu = np.zeros([2, NY, NX + 1])
        t_mu[0] = np.array([[2, 1], [2, 1], [2, 1]])
        t_mu[1] = np.array([[-2, -1], [-2, -1], [-2, -1]])
        t_mu_lam = np.concatenate((t_mu, t_lam), 2)
        t_Lam = np.concatenate((t_A, t_mu, t_lam), 2)  # 2 * NY * (NX + NXI + NETA +1)
        n_eta = np.concatenate((t_xi, x, t_eta), 2)  # N * NT * (1+ NX + NXI + NETA)
        y_mean[no_cure[0]] = np.matmul(t_Lam[0], n_eta[no_cure[0], :, :, np.newaxis]).squeeze(3)
        y_mean[cure[0]] = np.matmul(t_Lam[1], n_eta[cure[0], :, :, np.newaxis]).squeeze(3)
        y[no_cure[0]] = nrd.normal(y_mean[no_cure[0]], np.sqrt([t_phi[0]]))
        y[cure[0]] = nrd.normal(y_mean[cure[0]], np.sqrt([t_phi[1]]))
        y[no_obs[0], no_obs[1]] = 0
        # # ##======================define hazard function of t=================================================##
        f = np.zeros(shape=[N, NF])
        f[:, 0] = nrd.normal(0, 1, N)
        f[:, 1] = nrd.binomial(1, 0.5, N)
        t_psi = np.array([-1, -0.5])  ## coefficient for x, basline1, baseline2, and m
        t_tau = np.array([-0.5, -0.5, -0.5])
        t_pi = np.array([-0.5])
        mh = np.sum(f * t_psi, 1) + np.sum(t_xi[:, 0] * t_tau, 1) + t_w * t_pi
        t = Symbol('t')
        T = Symbol('T')

        # def hazard(t, mh):
        #     return (t / 2 + 0.5) * exp(mh)  ##---------------pay attention to here: we just assume mediator effect is time-varying----------------------------
        def hazard(t, mh):
            return 1 * exp(
                mh)  ##---------------pay attention to here: we just assume mediator effect is time-varying----------------------------

        def Hazard(hazard, T, mh):
            return scipy.integrate.quad(hazard, 0, T, args=mh)[0]

        def Inv_Hazard(Hazard, hazard, mh):
            U = -log(nrd.uniform(0, 1))

            def root_func(T):
                return U - Hazard(hazard, T, mh)

            return fsolve(root_func, [1])

        ET = np.zeros(shape=N)
        # ET = nrd.uniform(0, 3, N)
        for i in range(N):
            ET[i] = Inv_Hazard(Hazard, hazard, mh[i])
            # print("i is %d, ET is %.3f" %(i, ET[i]))
        # print(ET[no_cure])
        ET[cure] = 99999999999  # ########### important(set large value for cure population)
        # c_max = 10
        c_max = 13
        CT = nrd.uniform(0, c_max, N)
        OT = np.minimum(CT, ET)
        delta = (CT >= ET) + 0  # observed failure (dead) is 1
        censor_rate = 1 - np.sum(delta) / N
        oc = 1 - np.sum(delta[no_cure]) / NO
        # print("the censor rate, no cured censor rate for rep %d is %.3f, %.3f" % (r, censor_rate, oc))
        print("the censor rate, no cured censor rate for rep %d is %.3f, %.3f" % (rep, censor_rate, oc), flush=True,
              file=open("data.txt", "a"))
        ######################---------------------------------------This is for save data--------------------------###
        # all_d[r] = d
        # all_tC[r] = t_C
        # all_xi[r] = xi
        # all_tb[r] = t_b
        # all_x[r] = x
        # all_tw[r] = t_w
        # all_y[r] = y
        # all_OT[r] = OT
        # all_CT[r] = CT
        # all_ET[r] = ET
        # np.save(all_d, "all_d.npy")
        # np.save(all_tC, "all_tc.npy")
        # np.save(all_xi, "all_xi.npy")
        # np.save(all_tb, "all_tb.npy")
        # np.save(all_x, "all_x.npy")
        # np.save(all_tw, "all_tw.npy")
        # np.save(all_y, "all_y.npy")
        # np.save(all_OT, "all_OT.npy")
        # np.save(all_CT, "all_CT.npy")
        # np.save(all_ET, "all_ET.npy")
        # #----------------------------------------------------------Two parts------------------------------------------##
        # #---------------------------------------------------------Segment line---------------------------------------##
        ###################---------------------------------------This is for loading data--------------------------##
        # d = np.load("all_d1.npy")[rep]
        # t_C = np.load("all_tc1.npy")[rep]
        # xi = np.load("all_xi1.npy")[rep]
        # t_eta = np.load("all_teta1.npy")[rep]
        # x = np.load("all_x1.npy")[rep]
        # y = np.load("all_y1.npy")[rep]
        # f = np.load("all_f1.npy")[rep]
        # t_w = np.load("all_tw1.npy")[rep]
        # OT = np.load("all_OT1.npy")[rep]
        # CT = np.load("all_CT1.npy")[rep]
        # ET = np.load("all_ET1.npy")[rep]
        # delta = (CT >= ET) + 0  # observed failure (dead) is 1
        #---------------------------------------------Set true parameters---------------------------------------
        # t_theta = np.array([-2, 1, 1])
        # t_alpha = np.array([-1, -1])
        # t_set = np.array([0, 0.01, 0.25, 0.5, 0.8, 1.5])
        # t_design = np.zeros(shape=[NT, NW])
        # t_design[:, 0] = 1
        # t_design[:, 1] = t_set
        # t_design[:, 2] = np.square(t_set)
        # t_beta = np.array([[1, 1], [0.5, 0.5]])
        # t_gamma = np.array([[1, 1], [0.5, 0.5]])
        # t_sigma = np.array([0.25, 0.36])
        # t_zeta = np.array([[0, 0, 0], [0, 0, 0]])
        # t_Sigma = np.array([[[1, 0.6, 0.6], [0.6, 1, 0.6], [0.6, 0.6, 1]],
        #                     [[1, 0.2, 0.2], [0.2, 1, 0.2], [0.2, 0.2, 1]]])  # 2 * NW * NW
        # t_mu = np.array([[1, 1, 1], [-1, -1, -1]])  # 2 * NY
        # t_lam = np.array([[[1], [0.8], [0.8]],
        #                   [[1], [0.9], [0.9]]])  # 2* NY * NETA
        # t_phi = np.array([[0.25, 0.25, 0.25], [0.36, 0.36, 0.36]])
        # t_Lam = np.concatenate((t_mu[:, :, np.newaxis], t_lam), 2)  # 2 * NY * (NETA +1)
        # t_psi = np.array([-1, -0.5])  ## coefficient for x, basline1, baseline2, and m
        # t_tau = np.array([-0.5, -0.5])
        # t_pi = np.array([-0.5, -0.5, -0.5])
        #--------------------Deal with observed number of covariates-------------------------------
        # T = np.sum(OT[:, np.newaxis] > t_set, 1)   # observed number of time
        # L = np.repeat(np.arange(1, NT+1)[np.newaxis, :], N, axis=0)
        # O = (L <= T[:, np.newaxis]).astype(int)   # observed or not due to death or censoring
        # OBS_T = np.sum(O, 1) #
        # no_obs = np.where(O == 0)
        # y[no_obs[0], no_obs[1]] = 99
        # oy = y.copy()
        #----------------------Initalize parameters--------------------------------------
        #---------------------------Factor analysis model--------------------
        lam = nrd.normal(0, 1, (2, NY, NXI + NX + NETA+1))
        lam[:, 0, NXI + NX + NETA] = 1
        phi = nrd.uniform(0, 1, (2, NY))
        Ind = np.ones(shape=[NY, NXI + NX + NETA+1])  # to show lambda is fixed or not (dim : NY * 3) fix is zero ( for each state is the same)
        Ind[0, NXI + NX + NETA] = 0     # fixed
        # #---------------------Trajectory model-----------------
        # tra = nrd.normal(0, 1, (2, NTRA))
        # t_tra = np.concatenate((t_beta, t_gamma, t_zeta), axis=1)
        # c_tra = np.array([0.1, 0.1])
        accept_tra = np.zeros(shape=[2])
        # sigma = nrd.uniform(0, 1, 2)
        # eta = nrd.normal(0, 1, (N, NT, NETA))
        #-----------------------Bspline model---------------------
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
        #------------------------------Replication--------------------
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
            w, accept_w = data.update_w(C, xi, y, lam, phi, x, f, sur, OT, lambd, nu, grid, delta, w, bs, Sigma, c_w, accept_w)
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
            if iter > 50 and iter % 50 == 0:
                print("%.3f seconds process time for one iter" % one_iter_time)
                print("%.3f seconds process time for one iter" % one_iter_time, flush=True, file=open("sur.txt", "a"))
                # print("Acceptance for sur, w and tra" % (accept_sur, accept_w, accept_tra), flush=True, file=open("sur.txt", "a"))
                rtime = Rep * Iter * one_iter_time * (1 - process) / 60
                print("Rep :%d, Iter : %d, process: %.3f, need %.1f min to complete" % (r, iter, process, rtime))
                print("acc of cured label for all and for censoring group is %.3f, %.3f" %(acc, acc_n))
                print("acc of cured label for all and for censoring group is %.3f, %.3f" % (acc, acc_n), flush=True,
                      file=open("sur.txt", "a"))
                print("Rep :%d, Iter : %d, process: %.3f, need %.1f min to complete" % (r, iter, process, rtime), flush=True,
                      file=open("sur.txt", "a"))
        m_bs = np.mean(all_bs[r, burnin:], 0)
        # mean = np.sum(m_bs * b_igt, 1)
        sp_b1 = BSpline(knots, m_bs[0], K)  # ---------------bspline for baseline-------------------------------
        # tt = np.reshape(t_set, (N*NT))[:500]
        tt = np.linspace(0, 10, 200)
        eft1 = sp_b1(tt) - np.mean(sp_b1(tt))
        # tft1 = np.reshape(t_design[0], (N*NT))[:500]
        # tft2 = np.reshape(t_design[1], (N*NT))[:500]
        tft1 = np.sqrt(tt) - np.mean(np.sqrt(tt))
        sp_b2 = BSpline(knots, m_bs[1], K)  # ---------------bspline for baseline-------------------------------
        eft2 = sp_b2(tt) - np.mean(sp_b2(tt))
        tft2 = np.log(tt + 1) - np.mean(np.log(tt + 1))
        plt.subplot(221)
        plt.plot(tt, eft1, lw=1, label='Estimated function')
        plt.plot(tt, tft1, lw=1, label='True function')
        plt.subplot(222)
        plt.plot(tt, eft2, lw=1, label='Estimated function')
        plt.plot(tt, tft2, lw=1, label='True function')
        e_w = np.mean(all_w[r, burnin:], 0)
        tttt = np.linspace(-4, 4, 200)
        plt.subplot(223)
        plt.scatter(t_w[no_cure[0]], e_w[no_cure[0]], lw=0.00001, label='Nocured')
        plt.plot(tttt, tttt, lw=1, label='line')
        plt.subplot(224)
        plt.scatter(t_w[cure[0]], e_w[cure[0]], lw=0.0001, label='Cured ')
        plt.plot(tttt, tttt, lw=1, label='line')
        plt.show()
        print("Please test")
    print("Replication is over")
    np.save("Sigma" + str(o) + ".npy", all_Sigma)
    np.save("w" + str(o) + ".npy", all_w)
    np.save("sur" + str(o) + ".npy", all_sur)
    np.save("lam" + str(o) + ".npy", all_lam)
    np.save("lambd" + str(o) + ".npy", all_lambd)
    np.save("phi" + str(o) + ".npy", all_phi)
    np.save("inc" + str(o) + ".npy", all_inc)
    np.save("C" + str(o) + ".npy", all_C)
    np.save("accept_sur" + str(o)+".npy",accept_sur)
    np.save("accept_w" + str(o)+".npy",accept_w)
    np.save("accept_tra" + str(o)+".npy",accept_tra)

if __name__ == '__main__':
    numList = []
    for o in range(0, 15):
        p = multiprocessing.Process(target=do, args=(o,))
        numList.append(p)
        p.start()
        p.join()

