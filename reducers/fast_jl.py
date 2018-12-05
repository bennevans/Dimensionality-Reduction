
from dim_reduction import DimensionReducer
import torch
import numpy as np

class FastJohnsonLindenstrauss(DimensionReducer):
    def __init__(self, d, d_prime, num_repeats=1):
        super(FastJohnsonLindenstrauss, self).__init__(d, d_prime)
        self.num_repeats = num_repeats
        self.new_dim = int(2**np.ceil((np.log2(self.d))))

    def construct(self, dataset):
        self.subsampling_indices = np.random.choice(self.new_dim, size=self.d_prime, replace=False)
        self.Ds = []
        for _ in range(self.num_repeats):
            self.Ds.append(np.random.choice([-1, 1], size=self.d))
    
    def save(self, dir):
        pass
        # np.save(dir, self.random_matrix.cpu().numpy())
    
    def load(self, dir):
        pass
        # self.random_matrix = torch.from_numpy(np.load(dir)).to(self.device)

    def fast_hadamard_transform_single(self, x):
        add_dim = self.new_dim - self.d
        x_power_2 = np.concatenate([x, np.zeros((add_dim))])
        bit = length = len(x_power_2)
        result = x_power_2.copy()
        for _ in range(int(np.log2(length))):
            bit >>= 1
            for i in range(length):
                if i & bit == 0:
                    j = i | bit
                    temp = result[i]
                    result[i] += result[j]
                    result[j] = temp - result[j]
        ret = result / np.sqrt(self.new_dim)
        return ret

    def fast_hadamard_transform(self, x):
        n = x.shape[0]
        d = x.shape[1]
        res = np.empty((n, self.new_dim))
        for i in range(x.shape[0]):
            res[i] = self.fast_hadamard_transform_single(x[i])
        return res

    def reduce_dim(self, x):
        for i in range(self.num_repeats):
            D = np.expand_dims(self.Ds[i], axis=0)
            rand_signed = x * D
            x = self.fast_hadamard_transform(rand_signed)
        return x[:, self.subsampling_indices] * np.sqrt(self.new_dim / self.d_prime)



from experiments.pairwise_distance import *
import matplotlib.pyplot as plt

np.warnings.filterwarnings('ignore')

if __name__ == '__main__':
    n = 300
    d = 1000
    d_prime = 200
    red = FastJohnsonLindenstrauss(d, d_prime, 1)
    data = np.random.randn(n, d).astype(np.float32)
    red.construct(data)
    red_data = red.reduce_dim(data)
    print('red_data', red_data.shape)

    pairwise_distances, pairwise_distances_reduced = create_pairwise_distances(data, red_data)
    average_absolute_epsilon = average_absolute_epsilon_from_pairwise_distances(pairwise_distances, pairwise_distances_reduced)
    print('average_absolute_epsilon', average_absolute_epsilon)
    eps = get_epsilons(pairwise_distances, pairwise_distances_reduced)
    print('std_dev', np.std(eps))
    print('mean', np.mean(eps))
    # plt.hist(eps)
    # plt.show()