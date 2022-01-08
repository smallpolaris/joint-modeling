from __future__ import division
from scipy.optimize import fsolve, root
from scipy import integrate
from sympy import Symbol, exp, log
from numpy import linalg as la
# from scipy.interpolate import BSpline
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
burnin = 4000
NX = 1
NY = 3
# NH = 3
NW = 1    ###### number of random effect for design time matrix
ND = 2
NF = 2
NXI = 5
NETA = 1  # number of  latent variabel
NG = 5  # number of grid
NK = 15  # number of inner knots
K = 2  # order (more explitily, degree (cubic)  (so number of spline is NK+K)
NB = NK + K - 1  # number of b-spline
NTRA = NX + NXI
NINC = ND + 1 + NXI
NSUR = NF + NXI + NW
Rep = 10
TNXI = 3
hh = 300
ww = 300
L = 30
#-------------------------To save data-------------------------
all_tw = np.zeros([Rep, N])
all_xi = np.zeros([Rep, N, NT, NXI])
all_d = np.zeros([Rep, N, ND + 1])
all_teta = np.zeros([Rep, N, NT, NETA])
all_x = np.zeros([Rep, N, NT, NX + 1])
all_f = np.zeros([Rep, N, NF])
all_ET = np.zeros(shape=[Rep, N])
all_CT = np.zeros(shape=[Rep, N])
all_OT = np.zeros(shape=[Rep, N])
all_delta = np.zeros(shape=[Rep, N])
all_tC = np.zeros(shape=[Rep, N])
all_y = np.zeros([Rep, N, NT, NY])
all_tset = np.zeros([Rep, N, NT])
all_vest_eimg = np.zeros([Rep, NXI, hh * ww])  # due to memory error, just save NXI eigenimg
all_est_ev = np.zeros([Rep, N*NT])  # save all eigenvalue
all_OBST = np.zeros([Rep, N])
all_obsnum = np.zeros([Rep])
for r in range(Rep):
    nrd.seed(r * 3 + 33)
    t0 = time.time()
    #############################---------------------------------FPCA(to be completed)-------------------------------#######
    #####_---------------------------------------Eigenimage----------------------------------##################
    t_eimg = np.zeros(shape=[TNXI, hh, ww])
    t_eimg[0, :100, :100] = 1
    t_eimg[1, 100:200, 100:200] = 1
    t_eimg[2, 200:300, 200:300] = 1
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
    print("SVD done in %r replication" % r, flush=True, file=open("sur.txt", "a"))
    obs_xi = np.matmul(np.diag(np.sqrt(s)),
                       ut)  # Estimated eigenscore   ( can truncated to   Trun* n  == eigenscore[:Trun]  Eigenscore)
    obs_xi = obs_xi[:NXI, :].transpose()  # Truncated to nxi
    # axi = np.matmul(np.diag(np.sqrt(s)), np.transpose(u))      xi and axi is the same
    pg = int(hh * ww / L)  # grid number of each part
    est_eimg = np.zeros(shape=[obs_num, L, pg])
    for l in range(L):
        y_part = obs_vimg[:, l * pg:(l + 1) * pg]
        est_eimg[:, l] = (np.matmul(y_part.transpose(),np.matmul(u, np.diag(1 / np.sqrt(s))))).transpose()  # eigenimage   [N * L * PG)
    vest_eimg = est_eimg.reshape((obs_num, -1))
    v_eimg = t_eimg.reshape((3, -1))  # vectorized value is also OK
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
    print("the censor rate, no cured censor rate for rep %d is %.3f, %.3f" % (r, censor_rate, oc), flush=True,
          file=open("data.txt", "a"))
    ######################---------------------------------------This is for save data--------------------------###
    all_d[r] = d
    all_tC[r] = t_C
    all_xi[r] = xi
    all_x[r] = x
    all_tw[r] = t_w
    all_teta[r] = t_eta
    all_OT[r] = OT
    all_CT[r] = CT
    all_ET[r] = ET
    all_f[r] = f
    all_y[r] = y
    all_tset[r] = t_set
    all_vest_eimg[r] = vest_eimg[:NXI]
    all_est_ev[r:, :obs_num] = est_ev
    all_OBST[r] = OBS_T
    all_obsnum[r] = obs_num
    print("Rep : %d" % r)
np.save("all_d.npy", all_d)
np.save("all_tc.npy", all_tC)
np.save("all_xi.npy", all_xi)
np.save("all_x.npy", all_x)
np.save("all_tw.npy", all_tw)
np.save("all_teta.npy", all_teta)
np.save("all_OT.npy", all_OT)
np.save("all_CT.npy", all_CT)
np.save("all_ET.npy", all_ET)
np.save("all_f.npy", all_f)
np.save("all_y.npy", all_y)
np.save("all_tset.npy", all_tset)
np.save("all_vest_eimg.npy", all_vest_eimg)
np.save("all_est_ev.npy", all_est_ev)
np.save("all_OBST.npy", all_OBST)
np.save("all_obsnum.npy", all_obsnum)

