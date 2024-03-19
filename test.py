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

            output = model(sample)
            _, predicted = torch.max(output.data, 1)

            total += target.size(0)
            correct += (predicted == target).sum()
    acc_test = float(correct) * 100 / total
    return acc_test


def valid_new(model, test_loader, sigma):
    model.eval()
    with torch.no_grad():
        correct, total = 0, 0
        for sample, target in test_loader:
            sample, target = sample.to(DEVICE).float(), target.to(DEVICE).long()
            sample = sample.view(-1, 9, 1, 128)

            sample = sample + torch.randn_like(sample, device='cuda') * sigma

            output, _ = model(sample)
            _, predicted = torch.max(output.data, 1)

            total += target.size(0)
            correct += (predicted == target).sum()
    acc_test = float(correct) * 100 / total
    return acc_test


def plot(data, args):
    plt.figure()

    # plt.plot(range(1, len(data[:, 0]) + 1),
    #          data[:, 0], color='blue', label='train')
    for i, sigma in enumerate(args.sigma_train):
        plt.plot(list(zip(*data[i]))[0], list(zip(*data[i]))[1], label=sigma)

    for i, sigma in enumerate(args.sigma_train):
        plt.plot(list(zip(*data[len(args.sigma_train) + i]))[0], list(zip(*data[len(args.sigma_train) + i]))[1], label=f"{sigma}_newloss")

    plt.legend()
    plt.xlabel('Sigma', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.title('Sigma and Test Accuracy', fontsize=16)

    plt.savefig(f"plot_new_loss.png")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, default='./')
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--seed', type=int, default=69)

    parser.add_argument('--sigma', type=int, default=2)
    parser.add_argument('--sigma_train', type=list, default=[0.0, 1.0,
                                                    2.0
                                                    ])
    parser.add_argument('--adv', type=str, default="Adv")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    torch.manual_seed(args.seed)

    train_loader, test_loader = data_preprocess.load(args.data_folder, batch_size=args.batchsize)
    
    
    model = net.Network().to(DEVICE)
    data = []

    for sigma_train in args.sigma_train:
        load_file = "logs/" + args.adv + f"/{sigma_train}/weight.pt"
        # load_file = "logs/" + "NewLoss" + f"/{sigma_train}/weight.pt"
        model = torch.load(load_file)
        sigma = 0
        tmp = []
        while sigma <= args.sigma:
            acc_test = valid(model, test_loader, sigma)
            print(f"Sigma {sigma}: {acc_test}")

            tmp.append([sigma, acc_test])

            sigma += 0.05
        
        data.append(tmp)

    for sigma_train in args.sigma_train:
        # load_file = "logs/" + args.adv + f"/{sigma_train}/weight.pt"
        load_file = "logs/" + "NewLoss" + f"/{sigma_train}/weight.pt"
        model = torch.load(load_file)
        sigma = 0
        tmp = []
        while sigma <= args.sigma:
            acc_test = valid(model, test_loader, sigma)
            print(f"Sigma {sigma}: {acc_test}")

            tmp.append([sigma, acc_test])

            sigma += 0.05
        
        data.append(tmp)

    # for mode in ["NewLossMoE", "MoE"]:
    #     load_file = "logs/" + mode + f"/weight.pt"
    #     model = torch.load(load_file)
    #     sigma = 0
    #     tmp = []
    #     while sigma <= args.sigma:
    #         acc_test = valid_new(model, test_loader, sigma)
    #         print(f"Sigma {sigma}: {acc_test}")

    #         tmp.append([sigma, acc_test])

    #         sigma += 0.05
        
    #     data.append(tmp)

    plot(data, args)

