#!/usr/bin/env python3

import argparse
import random
import pickle

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

import learn2learn as l2l
import torchvision.utils as vutils

import os

directory = './result'
if not os.path.exists(directory):
    os.makedirs(directory)
    
class Net(nn.Module):
    def __init__(self, latent_size):
        super(Net, self).__init__()
        self.latent_size = latent_size
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(True), 

            # nn.Linear(256, 128),
            # nn.BatchNorm1d(128),
            # nn.ReLU(True), 
        )
        
        self.mu = nn.Linear(64, latent_size)
        self.logvar = nn.Linear(64, latent_size)
        self.relu = nn.ReLU(True)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(True), 

            nn.Linear(64, 28*28),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, input_img):
        input_flat = input_img.view(-1, 28*28)
        x = self.encoder(input_flat)
        mu = self.relu(self.mu(x))
        logvar = self.relu(self.logvar(x))
        sample = self.reparameterize(mu, logvar)
        
        result = self.decoder(sample)
        
        # print(x[0][0].item(), mu[0][0].item(), logvar[0][0].item(), sample[0][0].item(), result[0][0].item())

        return result, mu, logvar

def save_result(adapt_x = None, adaptation_data=None, eval_x=None, evaluation_data=None, index=None):
    if index is None:
        index = 0

    if adaptation_data is not None:
        vutils.save_image(adaptation_data.view(-1, 1, 28, 28),
                './result/train_sup_' + str(index) + '.png',
                normalize=False)
    if adapt_x is not None:                
        vutils.save_image(adapt_x.view(-1, 1, 28, 28),
                './result/train_sup_t_' + str(index) + '.png',
                normalize=False)    
    if evaluation_data is not None:                                  
        vutils.save_image(evaluation_data.view(-1, 1, 28, 28),
                './result/train_qry_' + str(index) + '.png',
                normalize=False)
    if eval_x is not None:                                  
        vutils.save_image(eval_x.view(-1, 1, 28, 28),
                './result/train_qry_t_' + str(index) + '.png',
                normalize=False)      

def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1)
    acc = (predictions == targets).sum().float()
    acc /= len(targets)
    return acc.item()

def binary_cross_entropy(x, y, reduction='mean'):
    loss = -(x.log() * y + (1 - x).log() * (1 - y))
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()

def loss_func(recon_x, x, mu, log_var):
    BCE = binary_cross_entropy(recon_x, x.view(-1, 28*28), reduction='mean')
    KLD = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

    # print(BCE, KLD)
    return BCE + KLD*0.0015

def eval_model(meta_model, test_tasks, fas):
    learner = meta_model.clone()
    train_task = test_tasks.sample()
    data, labels = train_task
    data = data.to(device)
    labels = labels.to(device)

    loss = []
    for step in range(fas):
        # print(step)
        adapt_x, adapt_mu, adapt_log_var = learner(data)
        train_error = loss_func(adapt_x, data, adapt_mu, adapt_log_var)
        learner.adapt(train_error)
        loss.append(train_error.item()/len(data))
        # print(step, train_error.item()/len(adaptation_data))

        save_result(adapt_x = adapt_x, adaptation_data=data, eval_x=None, evaluation_data=None, index='testing')

    print('testing on 9:', loss)

    return learner.named_parameters()   

def save_gradients(base_param_values, param_values, index):
    param_names, param_values = [], []
    count = 0
    for name, param in param_values:
        if param.requires_grad:

            param_value = param.data.cpu().numpy() - base_param_values[count]
            param_values.append(param_value)
            param_names.append(name)

            print(param_value[0])
            print ('task', name, len(param.data.cpu().numpy()))

            count += 1

    # pickle.dump( param_names, open( "./result/param_name__" + str(t) + ".p", "wb" ) )
    pickle.dump(param_values, open( "./result/param_value__" + str(index) + ".p", "wb" ) )

def main(lr=0.005, maml_lr=0.01, iterations=1000, ways=1, shots=16, tps=32, fas=5, device=torch.device("cpu"),
         download_location='~/data'):
    transformations = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,)),
        lambda x: x.view(1, 28, 28),
    ])

    mnist_train = l2l.data.MetaDataset(MNIST(download_location,
                                             train=True,
                                             download=True,
                                             transform=transformations))

    train_tasks = l2l.data.TaskDataset(mnist_train,
                                       task_transforms=[
                                            l2l.data.transforms.FusedNWaysKShots(mnist_train, n=ways, k=shots*5, replacement=False, filter_labels=[0, 1, 2, 3, 4, 5, 6, 7, 8]),
                                            # l2l.data.transforms.NWays(mnist_train, ways),
                                            # l2l.data.transforms.KShots(mnist_train, shots*2),
                                            l2l.data.transforms.LoadData(mnist_train),
                                            l2l.data.transforms.RemapLabels(mnist_train),
                                            l2l.data.transforms.ConsecutiveLabels(mnist_train),
                                       ],
                                       num_tasks=1000)

    test_tasks = l2l.data.TaskDataset(mnist_train,
                                       task_transforms=[
                                            l2l.data.transforms.FusedNWaysKShots(mnist_train, n=ways, k=shots, replacement=False, filter_labels=[9]),
                                            # l2l.data.transforms.NWays(mnist_train, ways),
                                            # l2l.data.transforms.KShots(mnist_train, shots*2),
                                            l2l.data.transforms.LoadData(mnist_train),
                                            l2l.data.transforms.RemapLabels(mnist_train),
                                            l2l.data.transforms.ConsecutiveLabels(mnist_train),
                                       ],
                                       num_tasks=1000)

    model = Net(3)
    model.to(device)
    meta_model = l2l.algorithms.MetaSGD(model, lr=maml_lr)
    opt = optim.Adam(meta_model.parameters(), lr=lr)
    # loss_func = nn.NLLLoss(reduction='mean')

    # meta_model.load_state_dict(torch.load('./result/pre_trainedmodel'))

    for iteration in range(iterations):
        iteration_error = 0.0
        iteration_acc = 0.0
        losses = []
        for t in range(tps):
            # print('task', t)
            learner = meta_model.clone()
            train_task = train_tasks.sample()
            data, labels = train_task
            data = data.to(device)
            labels = labels.to(device)

            # Separate data into adaptation/evalutation sets
            adaptation_indices = np.zeros(data.size(0), dtype=bool)
            adaptation_indices[np.arange(shots*ways) * 1] = True
            evaluation_indices = torch.from_numpy(~adaptation_indices)
            adaptation_indices = torch.from_numpy(adaptation_indices)
            adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
            evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

            # print(adaptation_data.shape, evaluation_data.shape)

            # Fast Adaptation
            # print()
            loss = []
            adapt_x, eval_x = None, None
            for step in range(fas):
                # print(step)
                adapt_x, adapt_mu, adapt_log_var = learner(adaptation_data)
                train_error = loss_func(adapt_x, adaptation_data, adapt_mu, adapt_log_var)
                learner.adapt(train_error)
                loss.append(train_error.item()/len(adaptation_data))
                # print(step, train_error.item()/len(adaptation_data))
                if iteration%50 == 0 and (step==0):
                    save_result(adapt_x = adapt_x, adaptation_data=adaptation_data, eval_x=None, evaluation_data=None, index=t)

            if t == 0:
                print(loss)

            # save_gradients(base_param_values, learner.named_parameters(), t)
        
            # losses.append(loss)
            # Compute validation loss
            eval_x, eval_mu, eval_log_var = learner(evaluation_data)
            valid_error = loss_func(eval_x, evaluation_data, eval_mu, eval_log_var)

            valid_error /= len(evaluation_data)
            valid_accuracy = valid_error #accuracy(predictions, evaluation_labels)
            iteration_error += valid_error
            iteration_acc += valid_accuracy

            if iteration%50 == 0:
                save_result(adapt_x = adapt_x, adaptation_data=adaptation_data, eval_x=eval_x, evaluation_data=evaluation_data, index=str(t)+"_d")

        iteration_error /= tps
        iteration_acc /= tps
        print(iteration, 'Loss : {:.3f} Acc : {:.3f}'.format(iteration_error.item(), iteration_acc))

        if iteration%50 == 0:
            model_param = eval_model(meta_model, test_tasks, fas)
            # save_gradients(base_param_values, model_param, 'test')

        # Take the meta-learning step
        opt.zero_grad()
        iteration_error.backward()
        opt.step()

        if iteration%50 == 0:
            torch.save(meta_model.state_dict(), './result/model')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Learn2Learn MNIST Example')

    parser.add_argument('--ways', type=int, default=1, metavar='N',
                        help='number of ways (default: 5)')
    parser.add_argument('--shots', type=int, default=8, metavar='N',
                        help='number of shots (default: 1)')
    parser.add_argument('-tps', '--tasks-per-step', type=int, default=16, metavar='N',
                        help='tasks per step (default: 32)')
    parser.add_argument('-fas', '--fast-adaption-steps', type=int, default=5, metavar='N',
                        help='steps per fast adaption (default: 5)')

    parser.add_argument('--iterations', type=int, default=10000, metavar='N',
                        help='number of iterations (default: 1000)')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.005)')
    parser.add_argument('--maml-lr', type=float, default=1, metavar='LR',
                        help='learning rate for MAML (default: 0.01)')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    parser.add_argument('--download-location', type=str, default="/tmp/mnist", metavar='S',
                        help='download location for train data (default : /tmp/mnist')

    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if use_cuda else "cpu")

    main(lr=args.lr,
         maml_lr=args.maml_lr,
         iterations=args.iterations,
         ways=args.ways,
         shots=args.shots,
         tps=args.tasks_per_step,
         fas=args.fast_adaption_steps,
         device=device,
         download_location=args.download_location)
