# Main script to run for the Linear Model
# Use of the functions coded in homogfuncscale and global settings in homogsett
# The variables are denoted as in the main paper Section Numerical Experiments:
#    "Concentration Inequalities for Two-Sample Rank Processes with Application to Bipartite Ranking"

# author: Myrto Limnios // mail: myrto.limnios@ens-paris-saclay.fr

# What it does: outputs the optimal scoring parameter (theta star) for each of the statistic defined by the choice of the
# score-generating function (for example here MWW, RTB and Pol). For each one, it creates a csv file were the
# theta star is stored at every Monte-Carlo loop.
# Additionally, it outputs a csv file with the value of the statistic (loss) for each theta star.

import numpy as np
import pandas as pd
import datetime
from numpy.linalg import inv
import homogsett as sett
import time
import homogfunclin as func
import csv
from scipy.linalg import norm

start_time = time.time()
test_type = 'Loc'
random.seed(40)

### Dimension of the data
n =
m =
d =

### Discrepancy parameter
epsilon =

### Parameters of the GA algorithm
ratio_test =
B =   # number of MC simulations
T =   # number of ascent steps


dict_res_RTB, dict_res_Pol, dict_res_MWW = {}, {}, {}
dict_res_loss = {}

###  List of the methods for the ranking criteria (MWW: Mann-Whitney-Wilcoxon, Pol: Polynomial, RTB: Ranking the Best)
meth = ['MWW', 'Pol', 'RTB']

###  Complete with the path to your directory
path =

###  Initialize the parameters
str_param = str(n) + str(m) + str(d) + str(int(epsilon * 100)) + str(T)
sett.init(d, epsilon, test_type)

meanX = sett.meanX
meanY = sett.meanY
varX = sett.varX
varY = sett.varY

###  Compute the star parameters
orig_star = - meanX.T @ inv(varX) @ meanX + meanY.T @ inv(varX) @ meanY
theta_star = np.dot(inv(varX), (meanX - meanY)) / np.sqrt(norm(np.dot(inv(varX), (meanX - meanY))))

dict_res_RTB[str('star')] = theta_star
dict_res_MWW[str('star')] = theta_star
dict_res_Pol[str('star')] = theta_star

for i in meth:
    dict_res_loss[str(i) + str('loss_') + str_param] = np.zeros(T + 1)

theta_RTB_, theta_Pol_, theta_MWW_ = np.zeros(d), np.zeros(d), np.zeros(d)
for i in range(B):

    X = np.float32(np.random.multivariate_normal(meanX, varX, n))
    Y = np.float32(np.random.multivariate_normal(meanY, varY, m))

    theta = np.random.rand(d)

    theta_RTB, loss_RTB = func.optim_naive_ga(func.scoring_RTB, func.scoring_RTB_d, X, Y, theta, T, eps=1e-3)
    dict_res_RTB[str('RTB_') + str(i)] = theta_RTB
    theta_RTB_ += theta_RTB / B

    theta_Pol, loss_Pol = func.optim_naive_ga(func.scoring_P, func.scoring_P_d, X, Y, theta, T, eps=1e-3)
    dict_res_Pol[str('Pol_') + str(i)] = theta_Pol
    theta_Pol_ += theta_Pol / B

    theta_MWW, loss_MWW = func.optim_naive_ga(func.scoring_MWW, func.scoring_MWW_d, X, Y, theta, T, eps=1e-3)
    dict_res_MWW[str('MWW_') + str(i)] = theta_MWW
    theta_MWW_ += theta_MWW / B


    loss_meth = [loss_MWW, loss_Pol, loss_RTB]

    for j in range(len(meth)):
        dict_res_loss[str(meth[j]) + str('loss_') + str_param] += np.divide(loss_meth[j], B)


dict_res_RTB[str('RTB_mean')] = theta_RTB_
dict_res_Pol[str('Pol_mean')] = theta_Pol_
dict_res_MWW[str('MWW_mean')] = theta_MWW_

df_res_RTB = pd.DataFrame(dict_res_RTB)
df_res_RTB.to_csv(path + 'resgenRTB_' + test_type + datetime.datetime.today().strftime("%m%d%H%M") + str_param + '.csv')
df_res_RTB = {}

df_res_Pol = pd.DataFrame(dict_res_Pol)
df_res_Pol.to_csv(path + 'resgenPol_' + test_type + datetime.datetime.today().strftime("%m%d%H%M") + str_param + '.csv')
df_res_Pol = {}

df_res_MWW = pd.DataFrame(dict_res_MWW)
df_res_MWW.to_csv(path + 'resgenMWW_' + test_type + datetime.datetime.today().strftime("%m%d%H%M") + str_param + '.csv')
df_res_MWW = {}

df_res_loss = pd.DataFrame(dict_res_loss)
df_res_loss.to_csv(path + 'resgenLoss_' + test_type + datetime.datetime.today().strftime("%m%d%H%M") + str_param + '.csv')
df_res_loss = {}


print("time elapsed: {:.2f}s".format(time.time() - start_time))


