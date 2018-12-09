
from experiments.pairwise_distance import *
from reducers.auto_encoder import AutoEncoder
from reducers.auto_encoder_norms import AutoEncoderNormRegularized
from reducers.jl import JohnsonLindenstrauss
from reducers.fast_jl import FastJohnsonLindenstrauss

import numpy as np
import time
import torch

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
                'd_primes': [8000, 4000, 2000, 1000, 500, 250, 100, 50, 10][::-1],
                # 'd_primes': [8000]
            },
            {
                'dataset_name': 'multimodal_features',
                'd': 15000,
                'd_primes': [10000, 8000, 4000, 2000, 500, 250, 100, 50, 10][::-1]
            },
            {
                'dataset_name': 'sparse_features',
                'd': 20000,
                'd_primes': [10, 50, 100, 250, 500, 1000, 2000, 4000, 8000, 16000]
            },
            {
                'dataset_name': 'ham_features',
                'd': 2352,
                'd_primes': [10, 50, 100, 250, 500, 1000, 2000]
            }
        ]
    }

def experiment(red_generator):
    exp_results = []
    pairwise_points = 1000
    for dataset in datasets_meta['datasets']:
        dataset_name = dataset['dataset_name']
        d = dataset['d']
        print('dataset_name', dataset_name, 'd', d, 'time', time.time())
        exp_data = data[dataset_name]
        for d_prime in dataset['d_primes']:
            print('\td_prime', d_prime)
            red = red_generator(d, d_prime)
            start = time.time()
            info = red.construct(exp_data)
            construct_time = time.time() - start
            print('\tinfo', info)
            start = time.time()
            red_data = red.reduce_dim(exp_data)
            reduce_all_time = time.time() - start
            print('\tconstruct time: {} reduce_time: {}'.format(construct_time, reduce_all_time))

            pairwise_distances, pairwise_distances_red = create_pairwise_distances(exp_data[:pairwise_points], red_data[:pairwise_points])
            avg_abs_eps = average_absolute_epsilon_from_pairwise_distances(pairwise_distances, pairwise_distances_red)
            epsilons = get_epsilons(pairwise_distances, pairwise_distances_red)

            exp_result = {
                'dataset': dataset_name,
                'd': d,
                'd_prime': d_prime,
                'avg_abs_eps': avg_abs_eps,
                'construct_time': construct_time,
                'reduce_time': reduce_all_time,
                'epsilons': epsilons
            }
            print('\t avg_abs_eps', avg_abs_eps)
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

def auto_encoder_exp():
    def red_generator(d, d_prime):
        dim1 = max(d//2, d_prime)
        dim2 = max(dim1//2, d_prime)
        sizes = [(d, dim1), (dim1, dim2), (dim2, d_prime)]
        print('sizes', sizes)
        non_linearity = torch.nn.Tanh
        mu = 1.0
        tau = 0.0
        self_norm_preservation = 0.1
        red = AutoEncoderNormRegularized(d, d_prime, sizes, non_linearity, mu, tau, self_norm_preservation,
            iterations=100000, batch_size=64, lr=1e-3, device=torch.device('cuda'), step_size=2500, gamma=0.8)
        return red
    return experiment(red_generator)

import pickle
save_prefix = '/raid/bevans97/datasets/cse599i/data/pairwise/'

if __name__ == '__main__':
    print('loading all data')
    data = load_data()
    print('starting experiments')
    
    # print('jl exp')
    # jl_exp_data = jl_exp()
    # print(jl_exp_data)
    # with open(save_prefix + 'jl_exp_data_sparse.p', 'wb') as f:
    #     pickle.dump(jl_exp_data, f)

    # print('fast jl exp')
    # fast_jl_exp_data = fast_jl_exp()
    # print(fast_jl_exp_data)
    # with open(save_prefix + 'fast_jl_sparse.p', 'wb') as f:
    #     pickle.dump(fast_jl_exp_data, f)

    # print('pca exp')
    # pca_data = pca_exp()
    # print(pca_data)
    # with open(save_prefix + 'pca_data_sparse.p', 'wb') as f:
    #     pickle.dump(pca_data, f)

    print('autoencoder exp')
    auto_encoder_exp_data = auto_encoder_exp()
    print(auto_encoder_exp_data)
    with open(save_prefix + 'auto_encoder_exp_data.p', 'wb') as f:
        pickle.dump(auto_encoder_exp_data, f)
    

