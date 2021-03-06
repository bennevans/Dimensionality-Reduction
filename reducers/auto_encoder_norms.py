
from reducers.dim_reduction import DimensionReducer
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Identity(nn.Module):
    def forward(self, x):
        return x

class AutoEncoderNormRegularized(DimensionReducer):

    def __init__(self, d, d_prime, sizes,
        non_linearity, mu, tau, self_norm, device=torch.device('cpu'), 
        batch_size=32, lr=1e-4, gamma=1.0,
        step_size=1000, iterations=5000,
        momentum=0.9, optimizer='adam'):
        super(AutoEncoderNormRegularized, self).__init__(d, d_prime)

        self.mu = mu
        self.tau = tau
        self.self_norm = self_norm
        self.device = device
        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma
        self.step_size = step_size
        self.iterations = iterations

        assert sizes[0][0] == d
        assert sizes[-1][1] == d_prime

        encoder_modules = []
        decoder_modules = []

        for in_size, out_size in sizes:
            encoder_modules.append(nn.Linear(in_size, out_size, bias=True))
            encoder_modules.append(non_linearity())
        for in_size, out_size in sizes[::-1]:
            decoder_modules.append(nn.Linear(out_size, in_size, bias=True))
            decoder_modules.append(non_linearity())
        encoder_modules[-1] = Identity()
        decoder_modules[-1] = Identity()

        print('modules')
        print(encoder_modules)
        print(decoder_modules)
        
        self.encoder = nn.Sequential(*encoder_modules).to(self.device)
        self.decoder = nn.Sequential(*decoder_modules).to(self.device)
        self.model = nn.Sequential(self.encoder, self.decoder).to(self.device)

        if optimizer == 'adam':
            self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer == 'sgd':
            self.optim = torch.optim.SGD(self.model.parameters(), lr = lr, momentum=momentum)
        else:
            raise NotImplementedError('bad optimizer type: {}'.format(optimizer))

        self.sched = torch.optim.lr_scheduler.StepLR(self.optim, self.step_size, self.gamma)

    def norm(self, x, dim=1):
        return torch.sum(x**2, dim=dim)

    def encode(self, x):
        code = self.encoder(x) * np.sqrt(self.d / self.d_prime)
        return code
    def construct(self, data, print_interval=500, return_info=True):
        self.model.train()
        assert data.shape[1] == self.d
        n = data.shape[0]
        for i in range(self.iterations):
            self.sched.step()
            self.optim.zero_grad()
            batch_idx = np.random.choice(n, self.batch_size, replace=False)
            batch = torch.from_numpy(data[batch_idx]).to(self.device)

            code = self.encode(batch)
            predicted = self.decoder(code)

            reconstruction_loss = F.mse_loss(predicted, batch)
            pairwise_idx_1 = np.random.choice(self.batch_size, self.batch_size // 2)
            pairwise_idx_2 = np.random.choice(self.batch_size, self.batch_size // 2)
            true_dists = self.norm(batch[pairwise_idx_1] - batch[pairwise_idx_2])
            pred_dists = self.norm(code[pairwise_idx_1] - code[pairwise_idx_2])
            
            norm_loss = self.norm(true_dists - pred_dists, dim=0) / (self.batch_size / 2)

            self_norm_loss = self.norm(self.norm(code) - self.norm(batch), dim=0) / (self.batch_size)

            loss = self.tau * reconstruction_loss + self.mu * norm_loss + self.self_norm * self_norm_loss
            loss.backward()

            self.optim.step()

            if print_interval is not None and i % print_interval == 0:
                print('i', i, 'loss', loss.item(), 'norm_loss', norm_loss.item(), 'self_norm_loss', self_norm_loss.item(), 'rec_loss', reconstruction_loss.item(), 'lr', self.sched.get_lr()[0])

        self.model.eval()

        # calculate total reconstruction error
        total_error = 0.0
        for batch_num in range(n // self.batch_size + 1):
            if batch_num * self.batch_size >= n:
                break
            batch = torch.from_numpy(data[self.batch_size*batch_num:self.batch_size*(batch_num+1)]).to(self.device)
            predicted = self.model(batch)
            total_error += F.mse_loss(predicted, batch, reduction='sum').item()
        
        print('total_error', total_error)
        print('avg_error', total_error / n)

    def reduce_dim(self, data):
        return self.encode(torch.from_numpy(data).to(self.device)).cpu().detach().numpy()


from experiments.pairwise_distance import *
import matplotlib.pyplot as plt
# seed = 0xBEEF
seed = 0xBEED
np.random.seed(seed)
torch.manual_seed(seed)
import matplotlib.pyplot as plt

def linear_pca():
    d = 2
    d_prime = 1
    n = 100
    sizes = [(d, d_prime)]
    non_linearity = Identity

    # was just curious if non-liner nn would train easier. for some reason it does!
    # sizes = [(d, 4), (4,4), (4, d_prime)]
    # non_linearity = nn.Sigmoid

    data = np.array([[1,1],[-1,-1],[-0.2, 0.2], [0.2,-0.2]]).astype(np.float32)
    # data = np.random.multivariate_normal(np.zeros(2), np.array([[1,1],[1,0]]), size=(n)).astype(np.float32)

    # params for SGD
    # red = AutoEncoder(d, d_prime, sizes, non_linearity, iterations=10000, batch_size=4, lr=2e-3, step_size=10, gamma=0.99) # without momentum good params
    # red = AutoEncoder(d, d_prime, sizes, non_linearity, iterations=10000, batch_size=4, lr=1e-6, step_size=10, gamma=0.995) # pretty good with momentum
    # red = AutoEncoder(d, d_prime, sizes, non_linearity, iterations=50000, batch_size=4, lr=1e-6, step_size=20, gamma=0.995, momentum=0.6) # 0.0665
    # red = AutoEncoder(d, d_prime, sizes, non_linearity, iterations=50000, batch_size=4, lr=5e-7, step_size=50, gamma=0.995, momentum=0.3) # 0.052
    # red = AutoEncoder(d, d_prime, sizes, non_linearity, iterations=100000, batch_size=4, lr=5e-7, step_size=50, gamma=0.995, momentum=0.3) # 0.0456

    # Adam
    # red = AutoEncoder(d, d_prime, sizes, non_linearity, iterations=20000, batch_size=4, lr=1e-3, step_size=500, gamma=0.9) # 0.0445 better already


    red.construct(data, print_interval=100)
    
    xs = data[:,0]
    ys = data[:,1]
    red_data = red.reduce_dim(data)
    plt.scatter(xs, red_data)
    plt.scatter(xs, ys)
    plt.show()

    print(list(red.model.parameters()))

def non_linear_pca():
    d = 2
    d_prime = 1
    n = 100
    sizes = [(d, 10), (10, 10), (10, d_prime)]
    non_linearity = nn.Sigmoid

    xs = np.linspace(-1,1).astype(np.float32)
    ys = xs**2
    data = np.stack([xs,ys], axis=1)
    # data = np.random.multivariate_normal(np.zeros(2), np.array([[1,1],[1,0]]), size=(n)).astype(np.float32)

    red = AutoEncoderNormRegularized(d, d_prime, sizes, non_linearity, mu=1.0, tau=0.0, iterations=10000, batch_size=16, lr=1e-4, step_size=1000, gamma=1.0, optimizer='adam')
    red.construct(data, print_interval=1000)

    red_data = red.reduce_dim(data)
    # print(xs.shape, red_data.shape)
    # plt.scatter(xs, red_data)
    # plt.scatter(xs, ys)
    # plt.show()

    pairwise_distances, pairwise_distances_reduced = create_pairwise_distances(data, red_data)
    average_absolute_epsilon = average_absolute_epsilon_from_pairwise_distances(pairwise_distances, pairwise_distances_reduced)
    print('average_absolute_epsilon', average_absolute_epsilon)
    eps = get_epsilons(pairwise_distances, pairwise_distances_reduced)
    print('std_dev', np.std(eps))
    print('mean', np.mean(eps))
    plt.hist(eps)
    plt.show()

if __name__ == '__main__':
    non_linear_pca()
    # linear_pca()