import numpy as np
import scipy.linalg
import argparse
import os
import torch
from torch.distributions.multivariate_normal import MultivariateNormal


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(75)
    n, d, m = args.n, args.d, args.m

    labels_list = []
    data_list = []
    means_list = []
    covs_list = []

    l=1

    for i in range(m):

        mean = torch.randn(d)*2
        covs = torch.randn((d,d))
        covs = covs @ covs.t() + l*torch.eye(d)

        means_list.append(mean)
        covs_list.append(covs)

        sampler = MultivariateNormal(mean, covs)

        data = sampler.sample(sample_shape=(torch.Size([n // m])))
        labels = torch.zeros((n // m, m))
        labels[:, i] = 1.0
        labels_list.append(labels)
        data_list.append(data)
    
    data = torch.cat(data_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    means = torch.stack(means_list)
    covs = torch.stack(covs_list)

    print('data shape', data.shape)
    print('labels shape', labels.shape)

    path = os.path.join(args.save_prefix, 'multimodal_gaussian_' + str(n) + '_' + str(d) + '_' + str(m))
    os.makedirs(path)

    data_path = os.path.join(path, 'data.npy')
    labels_path = os.path.join(path, 'labels.npy')
    means_path = os.path.join(path, 'means.npy')
    covs_path = os.path.join(path, 'covs.npy')

    np.save(data_path, data.numpy())
    np.save(labels_path, labels.numpy())
    np.save(means_path, means.numpy())
    np.save(covs_path, covs.numpy())
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('n', type=int)
    parser.add_argument('d', type=int)
    parser.add_argument('m', type=int, help='number of modes')
    parser.add_argument('--save-prefix', default='/raid/bevans97/datasets/cse599i', type=str)
    parser.add_argument('--seed', default=0xDEADBEEF, type=int)
    args = parser.parse_args()
    main(args)
