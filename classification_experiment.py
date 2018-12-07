
from experiments.classification import train_and_score
import numpy as np

from reducers.auto_encoder import AutoEncoder
from reducers.auto_encoder_norms import AutoEncoderNormRegularized
from reducers.jl import JohnsonLindenstrauss
from reducers.fast_jl import FastJohnsonLindenstrauss

import torch

datasets_prefix = '/raid/bevans97/datasets/cse599i'
data_dir = '/data.npy'
labels_dir = '/labels.npy'

multimodal_gaussian_suffix = datasets_prefix+'/multimodal_gaussian_20000_15000_2'

def load_multimodal():
    data = np.load(multimodal_gaussian_suffix + data_dir)
    labels = np.load(multimodal_gaussian_suffix + labels_dir)
    return data, labels

d_primes = [10, 50, 100, 250, 500, 1000, 2000, 4000, 8000]
d = 15000

multimodal_data, multimodal_labels = load_multimodal()
multimodal_labels = np.argwhere(multimodal_labels==1)[:,1]

def experiment(red_generator):
    exp_results = []
    print('multimodal_data', multimodal_data.shape)
    print('multmodal_lables', multimodal_labels.shape)
    score = train_and_score(multimodal_data, multimodal_labels)
    for d_prime in d_primes:
        red = red_generator(d, d_prime)
        red.construct(multimodal_data)
        red_data = red.reduce_dim(multimodal_data)

        red_score = train_and_score(red_data, multimodal_labels)

        exp_result = {
            'd': d,
            'd_prime': d_prime,
            'red_score': red_score,
            'score': score
        }

        exp_results.append(exp_result.copy())
    return {'data': exp_results}

def jl_exp():
    red_generator = JohnsonLindenstrauss
    return experiment(red_generator)

def fast_jl_exp():
    red_generator = FastJohnsonLindenstrauss
    return experiment(red_generator)

from reducers.auto_encoder import Identity

def pca_exp():
    def red_generator(d, d_prime):
        sizes = [(d, d_prime)]
        non_linearity = Identity
        red = AutoEncoder(d, d_prime, sizes, non_linearity,
            iterations=10000, batch_size=256, lr=1e-3, device=torch.device('cuda'), step_size=2500, gamma=0.75)
        return red
    return experiment(red_generator)


import pickle
save_prefix = '/raid/bevans97/datasets/cse599i/data/classif/'

if __name__ == '__main__':
    print('loading all data')
    print('starting experiments')
    print('jl exp')

    jl_exp_data = jl_exp()
    print(jl_exp_data)
    
    with open(save_prefix + 'jl_exp_data.p', 'wb') as f:
        pickle.dump(jl_exp_data, f)

    print('fast jl exp')
    fast_jl_exp_data = fast_jl_exp()
    print(fast_jl_exp_data)
    with open(save_prefix + 'fast_jl_exp_data.p', 'wb') as f:
        pickle.dump(fast_jl_exp_data, f)

    print('pca exp')
    pca_data = pca_exp()
    print(pca_data)
    with open(save_prefix + 'pca_data.p', 'wb') as f:
        pickle.dump(pca_data, f)

