
from reducers.dim_reduction import DimensionReducer
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

    # def fast_hadamard_transform(self, x):
    #     n = x.shape[0]
    #     add_dim = self.new_dim - self.d
    #     x_power_2 = np.concatenate([x, np.zeros((n,add_dim))], axis=1)
    #     bit = length = self.new_dim
    #     print('x_power_2', x_power_2.shape, x_power_2.dtype)
    #     result = x_power_2.copy()

    #     for _ in range(int(np.log2(length))):
    #         bit >>= 1
    #         for i in range(length):
    #             if i & bit == 0:
    #                 j = i | bit
    #                 temp = result[:, i].copy()
    #                 result[:, i] += result[:, j]
    #                 result[:,j] = temp - result[:,j]

    #     result /= np.sqrt(self.new_dim)
    #     return result

    def fast_hadamard_transform_pytorch(self, x, device=torch.device('cuda')):
        n = x.shape[0]
        add_dim = self.new_dim - self.d
        x_power_2 = np.concatenate([x, np.zeros((n,add_dim))], axis=1)
        bit = length = self.new_dim
        result = torch.tensor(x_power_2.copy().astype(np.float32)).to(device)

        for _ in range(int(np.log2(length))):
            bit >>= 1
            for i in range(length):
                if i & bit == 0:
                    j = i | bit
                    temp = result[:, i].clone()
                    result[:, i] += result[:, j]
                    result[:,j] = temp - result[:,j]

        result /= np.sqrt(self.new_dim)
        return result.cpu().numpy()

    def reduce_dim(self, x):
        for i in range(self.num_repeats):
            D = np.expand_dims(self.Ds[i], axis=0)
            rand_signed = x * D
            x = self.fast_hadamard_transform_pytorch(rand_signed)
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
    plt.hist(eps)
    plt.show()