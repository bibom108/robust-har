from itertools import accumulate
import data_preprocess
import matplotlib.pyplot as plt
import models.network as net
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
result = []


def valid(model, test_loader, sigma):
    model.eval()
    with torch.no_grad():
        correct, total = 0, 0
        for sample, target in test_loader:
            sample, target = sample.to(DEVICE).float(), target.to(DEVICE).long()
            sample = sample.view(-1, 9, 1, 128)

            sample = sample + torch.randn_like(sample, device='cuda') * sigma

            # output, _ = model(sample)
            output = model(sample)

            _, predicted = torch.max(output.data, 1)

            total += target.size(0)
            correct += (predicted == target).sum()
    acc_test = float(correct) * 100 / total
    return acc_test


def plot(data):
    plt.figure()

    # plt.plot(range(1, len(data[:, 0]) + 1),
    #          data[:, 0], color='blue', label='train')
    
    plt.plot(list(zip(*data))[0], list(zip(*data))[1])

    plt.legend()
    plt.xlabel('Sigma', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.title('Sigma and Test Accuracy', fontsize=16)

    plt.savefig(f"simple_test_plot.png")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, default='./')
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--seed', type=int, default=69)
    parser.add_argument('--sigma_train', type=float, default=0)
    parser.add_argument('--sigma', type=float, default=2)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    torch.manual_seed(args.seed)

    train_loader, test_loader = data_preprocess.load(args.data_folder, batch_size=args.batchsize)
    
    
    model = torch.load(f"logs/NewLoss/0.5/weight.pt")
    sigma = 0
    data = []
    while sigma <= args.sigma:
        acc_test = valid(model, test_loader, sigma)
        print(f"Sigma {sigma}: {acc_test}")
        data.append([sigma, acc_test])
        sigma += 0.05
    plot(data)

