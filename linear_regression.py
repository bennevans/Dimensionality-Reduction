import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.utils.data
import yaml
import time
import os

class LinearModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearModel, self).__init__()

        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        features, labels = sample['feature'], sample['label']

        return {'feature': torch.from_numpy(features),
                'label': torch.from_numpy(labels)}

class NumpyDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_prefix, transform=None):
        self.data = np.load(dataset_prefix+'data.npy')
        self.labels = np.expand_dims(np.load(dataset_prefix+'labels.npy'), axis=1)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {'feature': self.data[idx], 'label': self.labels[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample

def main():
    prefix = '/raid/bevans97/datasets/cse599i/'
    dataset_prefix = prefix + 'sparse_100000_20000/'
    
    batch_size = 2**12
    lr = 0.02
    num_epochs = 50
    log_interval = 10
    step_size = 1
    gamma = 0.75
    device = torch.device('cuda')

    print('loading data')
    dataset = NumpyDataset(dataset_prefix, transform=ToTensor())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    data_dim = dataset.data.shape[1]

    model = LinearModel(data_dim).to(device)
    optim = torch.optim.SGD(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(optim, step_size, gamma)

    print('training')
    train_info = {
        'loss': [],
        'lr': [],
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'step_size': step_size,
        'gamma': gamma,
        'data_dim': data_dim
    }

    total_batches = 0

    for epoch in range(num_epochs):
        sched.step()
        for batch in dataloader:
            optim.zero_grad()
            total_batches += 1
            inputs = batch['feature'].to(device)
            labels = batch['label'].to(device)

            predicted = model(inputs)
            loss = F.mse_loss(predicted, labels)
            loss.backward()
            optim.step()

            if total_batches % log_interval == 0:
                train_info['loss'].append(loss.item())
                train_info['lr'].append(sched.get_lr()[0])
                print('epoch {}, batches {}, loss {}, lr {}'.format(epoch, total_batches, loss, sched.get_lr()[0]))
        print('epoch {}, batches {}, loss {}, lr {}'.format(epoch, total_batches, loss, sched.get_lr()[0]))

    print('done training!')
    now = time.strftime('%c')
    log_dir = prefix + 'log/' + now + '/'
    os.makedirs(log_dir)
    torch.save(model.state_dict(), log_dir + 'params.pt')
    with open(log_dir + 'metadata.yml', 'w') as f:
        yaml.dump(train_info, f, default_flow_style=False)

if __name__ == '__main__':
    main()