# Script with all the functions used for the Gradient Ascent Algorithm for the Linear Model
# The variables are denoted as in the main paper Section Numerical Experiments:
#    "Concentration Inequalities for Two-Sample Rank Processes with Application to Bipartite Ranking"

# author: Myrto Limnios // mail: myrto.limnios@ens-paris-saclay.fr

import numpy as np
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split
import scipy.stats as stats

from sklearn.preprocessing import StandardScaler
from scipy.linalg import norm

seed_num = 40
scaler = StandardScaler()


def simu_data(meanX, meanY, varX, varY, n, m, ratio_test):

    X = np.float32(np.random.multivariate_normal(meanX, varX, n))
    Y = np.float32(np.random.multivariate_normal(meanY, varY, m))

    X, Xt = train_test_split(X, train_size=ratio_test, random_state=seed_num)
    Y, Yt = train_test_split(Y, train_size=ratio_test, random_state=seed_num)

    return X, Y, Xt, Yt


#### Score-generating function

def sftplus(x,beta = 1e2):
    return (1/beta) * np.log(1 + np.exp(beta*x))

def sig(x):
    return 1 / (1 + np.exp(-x))

def scoring_RTB(rank_x, N):
    u0 = 0.7
    return - (sftplus(rank_x / (N + 1) - u0) + u0 * sig(1e2 * (rank_x / (N + 1)-u0)))

def scoring_RTBh(rank_x, N):
    u0 = 0.8
    return - (sftplus(rank_x / (N + 1) - u0) + u0 * sig(1e2 * (rank_x / (N + 1)-u0)))


def scoring_RTBs(rank_x, N):
    u0 = 0.6
    return - (sftplus(rank_x / (N + 1) - u0) + u0 * sig(1e2 * (rank_x / (N + 1)-u0)))


def scoring_MWW(rank_x, N):
    return - (rank_x / (N + 1))

def scoring_P(rank_x, N):
    p = 3
    return - (rank_x / (N + 1))**(p)

def scoring_Med(rank_x, N):
    return - np.sqrt(rank_x / (N + 1))



#### Score-generating derviative function

def sftplus_d(x,beta = 1e2):
    return (1 / beta) * (1 + beta * np.exp(beta*x)) / (1 + beta * np.exp(beta*x))

def sig_d(x):
    return np.exp(-x) / (1 + np.exp(-x))**2

def scoring_RTB_d(rank_x, N):
    u0 = 0.7
    return - (1 / (N+1)) * (sftplus_d(rank_x / (N + 1) - u0) + u0 * 1e2 * (1 / (N + 1)) * sig_d(1e2 * (rank_x / (N + 1)-u0)))

def scoring_RTBh_d(rank_x, N):
    u0 = 0.8
    return - (1 / (N+1)) * (sftplus_d(rank_x / (N + 1) - u0) + u0 * 1e2 * (1 / (N + 1)) * sig_d(1e2 * (rank_x / (N + 1)-u0)))

def scoring_RTBs_d(rank_x, N):
    u0 = 0.6
    return - (1 / (N+1)) * (sftplus_d(rank_x / (N + 1) - u0) + u0 * 1e2 * (1 / (N + 1)) * sig_d(1e2 * (rank_x / (N + 1)-u0)))

def scoring_MWW_d(rank_x, N):
    return - (1 / (N + 1))

def scoring_P_d(rank_x, N):
    p = 3
    return - ( p / (N + 1) ) * (rank_x / (N + 1))**(p-1)

def scoring_Med_d(rank_x, N):
    k = 0.5 / np.sqrt(N+1)
    return - k / np.sqrt(rank_x)



####### Fonction scoring

def scor_quadr(Z,theta):
    score = np.dot(np.dot(Z, theta), Z.transpose())

    if score.ndim >1:
        return np.diag(score)
    else:
        return score

def scor_lin(Z, theta):
    return np.dot(Z, theta).tolist()

def scor_cnst(Z):
    return Z


######## Reegularization functions

def kernel(t):
    return stats.norm.pdf(t)

def kappa(t):
    return stats.norm.cdf(t)


######## CDF Empirique and its gradient

def cdf_ker_emp(t, SX, SY, theta, N, h):

    Gn = np.sum([kappa((t-x)/h) for x in SX])
    Hm = np.sum([kappa((t-y)/h) for y in SY])

    cdf = Gn + Hm

    return (1/N) * cdf

def grad_cdf_ker_emp(T, SX, SY, TX, TY, theta, N, h):
    st = scor_lin(T, theta)
    tt = scor_cnst(T)

    grad_Gn = np.sum([2 * kernel((st - SX[i]) / h) * (tt - TX[i]) for i in range(len(SX))], axis = 0)
    grad_Hm = np.sum([2 * kernel((st - SY[j]) / h) * (tt - TY[j]) for j in range(len(SY))], axis = 0)

    grad_cdf = grad_Gn + grad_Hm

    return (1 / (N*h)) * grad_cdf


######## Objective function

def wstat_obj(scoring, X, Y, theta, N):
    SX = scor_lin(X, theta)
    SY = scor_lin(Y, theta)  ### size nx1

    pool_sample = np.concatenate([SX, SY])
    idx_sort = np.argsort(pool_sample)

    rank_pool = np.argsort(idx_sort)
    rankx = [(1 + rank_pool[i]) for i in range(len(SX))]

    loss = np.sum([scoring(rx, N) for rx in rankx])

    return loss


######## Gradient Ascent Algorithm

def optim_naive_ga(scoring, scoring_d, X, Y, theta, n_epoch, eps=1e-3):

    N = len(X) + len(Y)
    loop = int(len(X))
    gamma = 1 / np.sqrt(loop)
    h = (1 / N) ** (1 / 5)

    rank_X_star = np.arange(len(Y)+1, N+1, 1)

    W_star_th = - np.sum([scoring(rank_X_star[i], N) for i in range(len(X))])
    print('Wstar_th', W_star_th)

    loss = 0
    loss_ = [loss, loss]

    theta = theta / norm(theta)
    theta_ = [theta, theta]

    k = 1
    #while ((np.abs(W_star_th - loss) > eps) & (k < n_epoch)) or (k == 1):
    while (np.abs(W_star_th - loss) > eps) or (k < n_epoch) or (k == 1):

        SX = scor_lin(X, theta)
        SY = scor_lin(Y, theta) ### size nx1
        TX = np.array([scor_cnst(x) for x in X])
        TY = np.array([scor_cnst(y) for y in Y])  ### size nxd

        pool_sample = np.concatenate([SX,SY])
        idx_sort = np.argsort(pool_sample)
        rank_pool = np.argsort(idx_sort)
        rankx = [(1 + rank_pool[i]) for i in range(len(SX))]

        loss = - np.sum([scoring(rx, N) for rx in rankx])
        loss_.append(loss)

        F_empX = [cdf_ker_emp(x, SX, SY, theta, N, h) for x in SX]
        grad_F_empX = np.array([grad_cdf_ker_emp(x, SX, SY, TX, TY, theta, N, h) for x in X])

        phid_grad_F_empX = [scoring_d((N + 1) * f, N) for f in F_empX]
        grad_W = np.sum([(N + 1) * phid_grad_F_empX[i] * grad_F_empX[i] for i in range(len(phid_grad_F_empX))], axis = 0)

        if (k >=1) & (np.dot(grad_W, theta_[k] - theta_[k-1]) <= 0):
            thetat = theta - gamma * grad_W / np.sqrt(norm(grad_W))
            theta = thetat
        k += 1

        theta_.append(theta)

    thetastar = theta_[np.argmax(loss_)]
    wstatstar = np.max(loss_)

    if k < n_epoch:
        loss_ = np.concatenate((loss_, loss_[k - 2] * np.ones(n_epoch - k)))

    return thetastar, loss_

def res_auc(X,Y):
    label = np.concatenate((np.zeros(len(X)), np.ones(len(Y))), axis=0)
    fpr, tpr, thresholds = roc_curve(label, np.array(np.concatenate((X, Y), axis=0)))
    return auc(fpr, tpr)