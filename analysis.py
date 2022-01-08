from __future__ import division
from math import pi
from scipy.interpolate import BSpline
# from itertools import count
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import numpy.random as nrd
import tool
import update
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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
TNXI = 3
NXI = 3
NETA = 1  # number of  latent variabel
NG = 5  # number of grid
NK = 15  # number of inner knots
K = 2  # order (more explitily, degree (cubic)  (so number of spline is NK+K)
NB = NK + K - 1  # number of b-spline
NTRA = NX + NXI
NINC = ND + 1 + NXI
NSUR = NF + NXI + NW
hh = 300
ww = 300
Rep = 1
rep_n = 1
rep = int(Rep / rep_n) #replication nums once
knots = np.zeros(shape=[NK + 2 * K])
knots[K:(NK + K)] = np.linspace(0, 10, NK, endpoint=True)
knots[:K] = 0
# knots[NK + K:] = np.max(t_set)
knots[NK + K:] = 10
all_lam = np.zeros([Rep, Iter, 2, NY,NXI + NX + NETA + 1])
all_phi = np.zeros([Rep, Iter, 2, NY])
all_C = np.zeros(shape=[Rep, Iter, N])
all_inc = np.zeros([Rep, Iter, NINC])
all_lambd = np.zeros([Rep, Iter, NG])
all_sur = np.zeros([Rep, Iter, NF + NXI + NW])
all_Sigma = np.zeros([Rep, Iter, 2])
all_w = np.zeros([Rep, Iter, N])
all_bs = np.zeros([Rep, Iter, 2, NB])
all_tau = np.zeros([Rep, Iter, 2])
accept_w = np.zeros([N])
# t_set = np.load("all_tset.npy")
# all_tC = np.load("all_tC.npy")
# all_tw = np.load("all_tw.npy")
###----------------------------True parameters----------------------------------####
tool1 = tool.Tools(N, NT)
t_Sigma = np.array([1, 0.64])
t_theta = np.array([-2, 1, 1])
t_alpha = np.array([-1, -1, -1])
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
t_tau = np.array([-0.5, -0.5, -0.5])
t_pi = np.array([-0.5])
t_sur = np.concatenate((t_psi, t_tau, t_pi))
t_inc = np.concatenate((t_theta, t_alpha))
#--------------------------------------------For plot eigenimage-----------------------------------#
t_ev = np.array([1, 0.5, 0.25])
#----------------------------------Load data------------------------------
for o in range(rep_n):
    o1 = o
    all_sur[o * rep: (o+1) * rep] = np.load("sur" + str(o1) + ".npy")
    all_lam[o * rep: (o+1) * rep] = np.load("lam" + str(o1) + ".npy")
    all_Sigma[o * rep: (o+1) * rep] = np.load("Sigma" + str(o1) + ".npy")
    # all_w[o * rep: (o+1) * rep] = np.load("w" + str(o1) + ".npy")
    all_phi[o * rep: (o+1) * rep] = np.load("phi" + str(o1) + '.npy')
    # all_lambd[o * rep: (o+1) * rep] = np.load("lambd" + str(o1) + '.npy')
    all_inc[o * rep: (o+1) * rep] = np.load("inc" + str(o1) + '.npy')
    all_bs[o * rep: (o+1) * rep] = np.load("bs" + str(o1) + '.npy')
    # all_C[o * rep: (o+1) * rep] = np.load("C" + str(o1) + ".npy")
#--------------------Plot the eigenimage---------------
####--------------------Load the estimated eigenvalue and eigenimg-----------------------------
est_ev = np.load("all_est_ev.npy")[:Rep]
est_eimg = np.load("all_vest_eimg.npy")[:Rep]  # Rep * TRUN * DIM
# all_obsnum = np.load("all_obsnum.npy")[:Rep]
# mest_ev = est_ev.copy()  # modified ev
mest_eimg = est_eimg.copy()
#-------------------------True eigenimge-----------------------
t_eimg = np.zeros(shape=[TNXI, hh, ww])
# t_eimg[0, :10, :10] = 1
# t_eimg[1, 10:20, 10:20] = 1
# t_eimg[2, 20:30, 20:30] = 1
t_eimg[0, :100, :100] = 1
t_eimg[1, 100:200, 100:200] = 1
t_eimg[2, 200:300, 200:300] = 1
tv_eimg = np.reshape(t_eimg, (t_eimg.shape[0], -1))
#-----------------------------Plot the mean and CI of eimg------------------------------------##
#---------------------------------------Need to justify whether to change the sign of eigenimage/ eigenvalue -
for ll in range(TNXI):
    loc = np.where(np.sum(est_eimg[:, ll] * tv_eimg[ll], 1) <0)[0]
    mest_eimg[loc, ll] = -est_eimg[loc, ll]
    all_lam[loc, :, :, :,ll] = - all_lam[loc, :, :, :, ll]
    all_sur[loc, :, NF + ll] = - all_sur[loc, :, NF + ll]
    all_inc[loc, :, ND + 1 + ll] = -all_inc[loc, :, ND + 1 + ll]
m_sur = np.mean(all_sur[:, burnin:], 1)
m_Sigma = np.mean(all_Sigma[:, burnin:], 1)
m_phi = np.mean(all_phi[:, burnin:], 1)
m_lam = np.mean(all_lam[:, burnin:], 1)
m_lambd = np.mean(all_lambd[:, burnin:], 1)
m_inc = np.mean(all_inc[:, burnin:], 1)
m_bs = np.mean(all_bs[:, burnin:], 0)
mse_inc = tool1.hrep_rmse(all_inc, t_inc, burnin)
mse_sur = tool1.hrep_rmse(all_sur, t_sur, burnin)
mse_lam = tool1.hrep_rmse(all_lam, t_Lam, burnin)
mse_phi = tool1.hrep_rmse(all_phi, t_phi, burnin)
mse_Sigma = tool1.hrep_rmse(all_Sigma, t_Sigma, burnin)
knots = np.zeros(shape=[NK + 2 * K])
knots[K:(NK + K)] = np.linspace(0, 10, NK, endpoint=True)
knots[:K] = 0
# knots[NK + K:] = np.max(t_set)
knots[NK + K:] = 10
# #---------------Detect right replication-----------------------#
# aa = np.where(m_lam[:,0,1, NXI + NX + 1]>0)[0]
# bb = np.where(m_lam[aa,1, 1, NXI + NX + 1]>0)[0]
# cc = aa[bb][:100]
cc = 0
# ###----------------------------------Plot the eigenimage---------------------------------#
# ###--------------------Rescale---------------------------
# mest_eimg = mest_eimg[cc]
s_estimg = (mest_eimg - np.min(mest_eimg, 2)[:, :, np.newaxis]) / \
         (np.max(mest_eimg, 2)[:, :, np.newaxis] - np.min(mest_eimg, 2)[:, :, np.newaxis])
m_eimg = np.mean(s_estimg, 0)  #TRUN * DIM
ci_eimg_5 = np.percentile(s_estimg, 2.5, axis=0)
ci_eimg_95= np.percentile(s_estimg, 97.5, axis=0)
ci_img_5 = np.reshape(ci_eimg_5, (est_eimg.shape[1], hh, ww))
ci_img_95 = np.reshape(ci_eimg_95, (est_eimg.shape[1], hh, ww))
m_img = np.reshape(m_eimg, (est_eimg.shape[1], hh, ww))
i = 1
o = plt.cm.get_cmap('gray')
o_r = o.reversed()
fig = plt.figure(figsize=(12,16))
for r in range(TNXI):
    # plt.subplots(4,3,i)
    ax = fig.add_subplot(4,3,i)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(t_eimg[r], cmap=o_r)
    ax.invert_yaxis()
    plt.title("eigenimage %d"% i)
    i += 1
for r in range(TNXI):
    ax = fig.add_subplot(4,3,i)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(m_img[r], cmap =o_r)
    ax.invert_yaxis()
    i += 1
for r in range(TNXI):
    ax = fig.add_subplot(4,3,i)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(ci_img_5[r], cmap =o_r)
    ax.invert_yaxis()
    i += 1
for r in range(TNXI):
    ax = fig.add_subplot(4,3,i)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(ci_img_95[r], cmap =o_r)
    ax.invert_yaxis()
    i += 1
plt.show()
print("Bias of parameters in logistic model:")
print(np.mean(m_inc, 0) - t_inc)
print("RMS of parameters in logistic model:")
print(np.mean(mse_inc, 0))
print("Bias of parameters in CFA model:")
print(np.mean(m_lam,0)-t_Lam)
print(np.mean(m_phi,0)-t_phi)
print("RMS of parameters in CFA model:")
print(np.mean(mse_lam,0))
print(np.mean(mse_phi,0))
print("Bias of parameters in sigma:")
print(np.mean(m_Sigma, 0) - t_Sigma)
print("RMS of parameters in sigma:")
print(np.mean(mse_Sigma, 0))
print("Bias of parameters in PH model:")
print(np.mean(m_sur, 0) - t_sur)
print("RMS of parameters in PH model:")
print(np.mean(mse_sur,0))
# # # # ------------------------------Plot 95% CI in trajectory model---------------------------#  ##
# m_bs = np.mean(all_bs[cc, burnin:],(0,1))
ci_bs = np.zeros([Rep, 2, 2, NB])
tt = np.linspace(0, 10, 200)
all_f = np.zeros([Rep, Iter, 2, 200])
for r in range(Rep):
    for iter in range(Iter):
        sp0 = BSpline(knots, all_bs[r, iter, 0], K)
        all_f[r, iter, 0] = sp0(tt) - np.mean(sp0(tt))
        sp1 = BSpline(knots, all_bs[r, iter, 1], K)
        all_f[r, iter, 1] = sp1(tt) - np.mean(sp1(tt))
f_m = np.mean(all_f[:, burnin:], (0,1))
f_5 = np.percentile(all_f[:, burnin:], 2.5, axis=1)
f_95 = np.percentile(all_f[:, burnin:], 97.5, axis=1)
fm_5 = np.mean(f_5, axis=0)
fm_95 = np.mean(f_95, axis=0)
tft1 = np.sqrt(tt) - np.mean(np.sqrt(tt))
tft2 = np.log(tt + 1) - np.mean(np.log(tt + 1))
plt.figure(figsize=(10, 8))
plt.subplot(121)
plt.plot(tt, fm_5[0], lw=1, label='2.5% quantile', color="blue")
plt.plot(tt, f_m[0], lw=1, label='Estimated', color="red")
plt.plot(tt, fm_95[0], lw=1, label='97.5% quantile', color="green")
plt.plot(tt, tft1, lw=1, label="True", color="black")
plt.title("uncured group", fontsize='xx-large')
plt.ylim(-2.5, 1.5)
plt.subplot(122)
plt.plot(tt, fm_5[1], lw=1, label='2.5% quantile', color="blue")
plt.plot(tt, f_m[1], lw=1, label='Estimated', color="red")
plt.plot(tt, fm_95[1], lw=1, label='97.5% quantile', color="green")
plt.plot(tt, tft2, lw=1, label="True", color="black")
plt.title("cured group", fontsize='xx-large')
plt.legend()
plt.ylim(-2, 1)
plt.show()
print("Over")
# np.save("fm5.npy", fm_5)
# np.save("fm95.npy", fm_95)
