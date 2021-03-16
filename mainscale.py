# Main script to run for the Scale Model
# Use of the functions coded in homogfuncscale and global settings in homogsett
# The variables are denoted as in the main paper Section Numerical Experiments:
#    "Concentration Inequalities for Two-Sample Rank Processes with Application to Bipartite Ranking"

# author: Myrto Limnios // mail: myrto.limnios@ens-paris-saclay.fr

# What it does: outputs the optimal scoring parameter (theta star) for each of the statistic defined by the choice of the
# score-generating function (for example here MWW, RTB and Pol). For each one, it creates a txt file were the
# theta star is stored at every Monte-Carlo loop.
# Additionally, it outputs a txt file with the value of the statistic (loss) for each theta star.

import random
import numpy as np
import pandas as pd
import datetime
import homogsett as sett
import time
import homogfuncscale as func


random.seed(40)
start_time = time.time()

test_type = 'Scale'

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

###  List of the methods for the ranking criteria (MWW: Mann-Whitney-Wilcoxon, Pol: Polynomial, RTB: Ranking the Best)
meth = ['MWW', 'Pol', 'RTB']

###  Complete with the path to your directory
path =

###  Initialize the parameters
str_param = str(n) + str(m) + str(d) + str(int(epsilon * 100)) + str(T)
sett.init(d, epsilon, test_type)

meanY = sett.meanY
meanX = sett.meanX
varY = sett.varY
varX = sett.varX

eigva, eigve = np.linalg.eig(varY)

varY_txt = open(path + str_param + '_varY.txt', "w+")
np.savetxt(varY_txt, varY, delimiter=',')
varY_txt.close()

#### Compute theta star
theta_star = - np.linalg.inv(varX) + np.linalg.inv(varY)


dict_res_loss = {}

RTB_txt = open(path + 'resgenRTB_' + test_type + datetime.datetime.today().strftime("%m%d%H%M") + str_param + '.txt',"w+")
MWW_txt = open(path + 'resgenMWW_' + test_type + datetime.datetime.today().strftime("%m%d%H%M") + str_param + '.txt',"w+")
Pol_txt = open(path + 'resgenPol_' + test_type + datetime.datetime.today().strftime("%m%d%H%M") + str_param + '.txt',"w+")

for i in meth:
    dict_res_loss[str(i) + str('loss_') + str_param] = np.zeros(T + 1)

theta_RTB_, theta_Pol_, theta_MWW_ = np.zeros((d,d)), np.zeros((d,d)), np.zeros((d,d))
for i in range(B):

    X = np.float32(np.random.multivariate_normal(meanX, varX, n))
    Y = np.float32(np.random.multivariate_normal(meanY, varY, m))

    A = np.random.rand(d,d)
    theta = np.dot(A,A)

    theta_RTB, loss_RTB = func.optim_naive_ga(func.scoring_RTB, func.scoring_RTB_d, X, Y, theta, T, eps=1e-3)
    np.savetxt(RTB_txt,theta_RTB, delimiter=',', newline='\n' , header='loop'+ str(i))
    theta_RTB_ += theta_RTB

    theta_Pol, loss_Pol = func.optim_naive_ga(func.scoring_P, func.scoring_P_d, X, Y, theta, T, eps=1e-3)
    np.savetxt(Pol_txt,theta_Pol, delimiter=',', newline='\n' , header='loop'+ str(i))
    theta_Pol_ += theta_Pol

    theta_MWW, loss_MWW = func.optim_naive_ga(func.scoring_MWW, func.scoring_MWW_d, X, Y, theta, T, eps=1e-3)
    np.savetxt(MWW_txt,theta_MWW, delimiter=',', newline='\n' , header='loop'+ str(i))
    theta_MWW_ += theta_MWW

    loss_meth = [loss_MWW, loss_Pol, loss_RTB]

    for j in range(len(meth)):
        dict_res_loss[str(meth[j]) + str('loss_') + str_param] += np.divide(loss_meth[j], B)

    if i in [1, 2, 10, 20, 30, 40]:

        df_res_loss = pd.DataFrame(dict_res_loss)
        df_res_loss.to_csv(path + 'resLoss_' + test_type + datetime.datetime.today().strftime("%m%d%H%M") + str_param + str(i) + '.csv')
        df_res_loss = {}

np.savetxt(RTB_txt,np.divide(theta_RTB_,B), delimiter=',', newline='\n' , header='Mean')
np.savetxt(MWW_txt,np.divide(theta_MWW_,B), delimiter=',', newline='\n' , header='Mean')
np.savetxt(Pol_txt,np.divide(theta_Pol_,B), delimiter=',', newline='\n' , header='Mean')

np.savetxt(RTB_txt,theta_star, delimiter=',', newline='\n' , header='thetastar')
np.savetxt(MWW_txt,theta_star, delimiter=',', newline='\n' , header='thetastar')
np.savetxt(Pol_txt,theta_star, delimiter=',', newline='\n' , header='thetastar')

RTB_txt.close()
MWW_txt.close()
Pol_txt.close()

df_res_loss = pd.DataFrame(dict_res_loss)
df_res_loss.to_csv(path + 'resgenLoss_' + test_type + datetime.datetime.today().strftime("%m%d%H%M") + str_param + '.csv')
df_res_loss = {}

print("time elapsed: {:.2f}s".format(time.time() - start_time))
