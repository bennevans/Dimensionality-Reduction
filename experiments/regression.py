

import numpy as np
from sklearn.linear_model import SGDRegressor

def train_and_score(data, labels):
    reg = SGDRegressor(max_iter=1000,tol=1e-4)
    reg.fit(data, labels)
    score = reg.score(data, labels)
    return score


