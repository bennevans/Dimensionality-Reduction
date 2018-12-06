
from experiments.pairwise_distance import *
from reducers.auto_encoder import AutoEncoder
from reducers.auto_encoder_norms import AutoEncoderNormRegularized
from reducers.jl import JohnsonLindenstrauss
from reducers.fast_jl import FastJohnsonLindenstrauss

import numpy as np
import time

datasets_prefix = '/raid/bevans97/datasets/cse599i'
data_dir = '/data.npy'
labels_dir = '/labels.npy'

gaussian_suffix = datasets_prefix+'/gaussian_20000_10000'
multimodal_gaussian_suffix = datasets_prefix+'/multimodal_gaussian_20000_15000_2'
sparse_suffix = datasets_prefix+'/sparse_100000_20000'
ham_suffix = datasets_prefix+'/ham10000'
ham_d = 2352


def load_gaussian_dataset():
    data = np.load(gaussian_suffix + data_dir).astype(np.float32)
    return data
def load_multimodal():
    data = np.load(multimodal_gaussian_suffix + data_dir)
    labels = np.load(multimodal_gaussian_suffix + labels_dir)
    return data, labels
def load_sparse():
    data = np.load(sparse_suffix + data_dir)
    labels = np.load(sparse_suffix + labels_dir)
    return data, labels
def load_ham():
    data = np.load(ham_suffix + data_dir).astype(np.float32)
    labels = np.load(ham_suffix + labels_dir)
    return data, labels

def load_data():
    gaussian_features = load_gaussian_dataset()
    multimodal_features, multimodal_labels = load_multimodal()
    sparse_features, sparse_labels = load_sparse()
    ham_features, ham_labels = load_ham()

    return {
        'gaussian_features': gaussian_features,
        'multimodal_features': multimodal_features,
        'multimodal_labels': multimodal_labels,
        'sparse_features': sparse_features,
        'sparse_labels': sparse_labels,
        'ham_features': ham_features,
        'ham_labels': ham_labels
    }

datasets_meta = {
        'datasets':
        [
            {
                'dataset_name': 'gaussian_features',
                'd': 10000,
                'd_primes': [8000, 4000, 2000, 1000, 500, 250, 100, 50, 10]
            },
            {
                'dataset_name': 'multimodal_features',
                'd': 15000,
                'd_primes': [10000, 8000, 4000, 2000, 500, 250, 100, 50, 10]
            },
            {
                'dataset_name': 'sparse_features',
                'd': 20000,
                'd_primes': [16000, 8000, 4000, 2000, 1000, 500, 250, 100, 50, 10]
            },
            {
                'dataset_name': 'ham_features',
                'd': 2352,
                'd_primes': [2000, 1500, 1000, 500, 250, 100, 50, 10]
            }
        ]
    }

def jl_exp():
    for dataset in datasets_meta['datasets']:
        dataset_name = dataset['dataset_name']
        d = dataset['d']
        print('dataset_name', dataset_name, 'd', d, 'time', time.time())
        exp_data = data[dataset_name]
        for d_prime in dataset['d_primes']:
            print('\td_prime', d_prime)
            
            red = JohnsonLindenstrauss(d, d_prime)
            start = time.time()
            red.construct(exp_data)
            construct_time = time.time() - start
            start = time.time()
            red_data = red.reduce_dim(exp_data)
            reduce_all_time = time.time() - start
            print('\tconstruct time: {} reduce_time: {}'.format(construct_time, reduce_all_time))

def fast_jl_exp():
    for dataset in datasets_meta['datasets']:
        dataset_name = dataset['dataset_name']
        d = dataset['d']
        print('dataset_name', dataset_name, 'd', d, 'time', time.time())
        exp_data = data[dataset_name]
        for d_prime in dataset['d_primes']:
            print('\td_prime', d_prime)
            
            red = FastJohnsonLindenstrauss(d, d_prime)
            start = time.time()
            red.construct(exp_data)
            construct_time = time.time() - start
            start = time.time()
            red_data = red.reduce_dim(exp_data)
            reduce_all_time = time.time() - start
            print('\tconstruct time: {} reduce_time: {}'.format(construct_time, reduce_all_time))

if __name__ == '__main__':
    print('loading all data')
    data = load_data()
    print('starting experiments')
    fast_jl_exp()
    

