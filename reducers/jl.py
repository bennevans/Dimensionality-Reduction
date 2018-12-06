
from reducers.dim_reduction import DimensionReducer
import torch
import numpy as np

class JohnsonLindenstrauss(DimensionReducer):
    def __init__(self, d, d_prime, device=None):
        super(JohnsonLindenstrauss, self).__init__(d, d_prime)
        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device = device

    def construct(self, dataset):
        self.random_matrix = torch.randn((self.d, self.d_prime), device=self.device) / np.sqrt(self.d_prime)
    
    def save(self, dir):
        np.save(dir, self.random_matrix.cpu().numpy())
    
    def load(self, dir):
        self.random_matrix = torch.from_numpy(np.load(dir)).to(self.device)

    def reduce_dim(self, x):
        x = torch.from_numpy(x)
        return (x @ self.random_matrix).cpu().numpy()
    
from experiments.pairwise_distance import *
import matplotlib.pyplot as plt
if __name__ == '__main__':
    red = JohnsonLindenstrauss(100, 50)
    data = np.random.randn(256, 100).astype(np.float32)
    red.construct(data)
    red_data = red.reduce_dim(data)

    pairwise_distances, pairwise_distances_reduced = create_pairwise_distances(data, red_data)
    average_absolute_epsilon = average_absolute_epsilon_from_pairwise_distances(pairwise_distances, pairwise_distances_reduced)
    print('average_absolute_epsilon', average_absolute_epsilon)
    eps = get_epsilons(pairwise_distances, pairwise_distances_reduced)
    print('std_dev', np.std(eps))
    plt.hist(eps)
    plt.show()

    

