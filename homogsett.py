# Global parameters for generating two datasets for the Gradient Ascent Algorithm
# The variables are denoted as in the main paper Section Numerical Experiments:
#    "Concentration Inequalities for Two-Sample Rank Processes with Application to Bipartite Ranking"

# author: Myrto Limnios // mail: myrto.limnios@ens-paris-saclay.fr

import numpy as np
import numpy.random as rd
from scipy.linalg import norm

def init(d, param_test, test_type):

    global meanX, meanY, varX, varY
    if test_type == 'Loc':
        meanX = np.zeros(d)
        meanY = param_test * np.ones(d)
        A = np.random.rand(d,d)
        S = np.dot(A.T, A)
        varX = S / np.sqrt(norm(S))
        varY = varX

    if test_type == 'Scale':
        meanX = np.zeros(d)
        meanY = meanX
        Id = np.diag(np.ones(d))
        A = rd.randn(d, d)
        varY = Id + (param_test / d) * (A.T + A)
        varX = Id

