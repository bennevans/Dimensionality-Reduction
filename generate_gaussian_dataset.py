import numpy as np
import scipy.linalg
import argparse
import os
import torch
from torch.distributions.multivariate_normal import MultivariateNormal


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    n, d = args.n, args.d
    mean = torch.zeros(d)
    covs = torch.randn((d,d))
    covs = covs @ covs.t()

    sampler = MultivariateNormal(mean, covs)

    data = sampler.sample(sample_shape=(torch.Size([n])))
    print('data shape', data.shape)

    path = os.path.join(args.save_prefix, 'gaussian_' + str(n) + '_' + str(d))
    os.makedirs(path)
    data_path = os.path.join(path, 'data.npy')
    cov_path = os.path.join(path, 'cov.npy')
    np.save(data_path, data.numpy())
    np.save(cov_path, covs.numpy())
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('n', type=int)
    parser.add_argument('d', type=int)
    parser.add_argument('--save-prefix', default='/raid/bevans97/datasets/cse599i', type=str)
    parser.add_argument('--seed', default=0xDEADBEEF, type=int)
    args = parser.parse_args()
    main(args)
