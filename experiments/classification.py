

import numpy as np
from sklearn.linear_model import SGDClassifier

def train_and_score(data, labels):
    clf = SGDClassifier(max_iter=4000, tol=1e-4, penalty='none')
    clf.fit(data, labels)
    score = clf.score(data, labels)
    return score

def train_and_score_red(data, reduced_data, labels):
    return train_and_score(data, labels), train_and_score(reduced_data, labels)



