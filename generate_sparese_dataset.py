import numpy as np
import scipy.linalg
import argparse
import os
import torch
from torch.distributions.binomial import Binomial


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    n, d = args.n, args.d
    p = np.sqrt(d) / d
    probs = torch.ones(d) * p
    sampler = Binomial(1, probs)

    data = sampler.sample(sample_shape=(torch.Size([n])))

    # add some correlation
    offset = int(np.sqrt(d))
    for i in range(offset):
        data[:,i + offset] = data[:, i]

    weight = torch.randn((d))
    noise = torch.randn((n)) / 2

    labels = data @ weight + noise

    print('data shape', data.shape)

    path = os.path.join(args.save_prefix, 'sparse_' + str(n) + '_' + str(d))
    os.makedirs(path)
    data_path = os.path.join(path, 'data.npy')
    p_path = os.path.join(path, 'p.npy')
    labels_path = os.path.join(path, 'labels.npy')
    np.save(data_path, data.numpy())
    np.save(p_path, p)
    np.save(labels_path, labels.numpy())
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('n', type=int)
    parser.add_argument('d', type=int)
    parser.add_argument('--save-prefix', default='/raid/bevans97/datasets/cse599i', type=str)
    parser.add_argument('--seed', default=0xDEADBEEF, type=int)
    args = parser.parse_args()
    main(args)
