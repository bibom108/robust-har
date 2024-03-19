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
from attack import Attacker, PGD_L2, DDN


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



def requires_grad_(model:torch.nn.Module, requires_grad:bool) -> None:
    for param in model.parameters():
        param.requires_grad_(requires_grad)


def train(model, optimizer, train_loader, test_loader, args, sigma):
    criterion = nn.CrossEntropyLoss()

    # attacker = PGD_L2(steps=args.num_steps, device='cuda', max_norm=args.epsilon)
    attacker = PGD_L2(steps=args.num_steps, device='cuda', max_norm=sigma)

    result = []
    for e in range(args.nepoch):
        model.train()
        correct, total_loss = 0, 0
        total = 0
        for sample, target in train_loader:
            sample, target = sample.to(DEVICE).float(), target.to(DEVICE).long()
            sample = sample.view(-1, 9, 1, 128)


            noise = torch.randn_like(sample, device='cuda') * sigma

            if args.adv == "Adv":
                requires_grad_(model, False)
                model.eval()
                sample_w_noise = attacker.attack(model, sample, target, 
                                        noise=noise, 
                                        num_noise_vectors=args.num_noise_vec, 
                                        no_grad=args.no_grad_attack
                                        )
                model.train()
                requires_grad_(model, True)

            sample_w_noise = sample_w_noise + noise
            output_w_noise = model(sample_w_noise)
            output = model(sample)

            loss = criterion(output, target)
            loss_w_noise = criterion(output_w_noise, target)
            loss = loss + loss_w_noise

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum()
        acc_train = float(correct) * 100.0 / len(train_loader.dataset)

        # Testing
        acc_test = valid(model, test_loader)
        print(f'Epoch: [{e}/{args.nepoch}], loss:{total_loss / len(train_loader):.4f}, train_acc: {acc_train:.2f}, test_acc: {float(correct) * 100 / total:.2f}')
        result.append([acc_train, acc_test])
        # result_np = np.array(result, dtype=float)
        # np.savetxt('logs/result_np.csv', result_np, fmt='%.2f', delimiter=',')
    return result


def valid(model, test_loader):
    model.eval()
    with torch.no_grad():
        correct, total = 0, 0
        for sample, target in test_loader:
            sample, target = sample.to(DEVICE).float(), target.to(DEVICE).long()
            sample = sample.view(-1, 9, 1, 128)

            output = model(sample)
            _, predicted = torch.max(output.data, 1)

            total += target.size(0)
            correct += (predicted == target).sum()
    acc_test = float(correct) * 100 / total
    return acc_test


def plot(args):
    data = np.loadtxt(args.save_folder + "result.csv", delimiter=',')
    plt.figure()
    plt.plot(range(1, len(data[:, 0]) + 1),
             data[:, 0], color='blue', label='train')
    plt.plot(range(1, len(data[:, 1]) + 1),
             data[:, 1], color='red', label='test')
    plt.legend()
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.title('Train and Test Accuracy', fontsize=16)
    plt.savefig(args.save_folder + "train_plot.png")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nepoch', type=int, default=50)
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--lr', type=float, default=.01)
    parser.add_argument('--momentum', type=float, default=.9)
    parser.add_argument('--data_folder', type=str, default='./')
    parser.add_argument('--seed', type=int, default=69)


    parser.add_argument('--sigma', type=list, default=[0.0, 0.1, 0.2, 0.3, 0.4, 
                                                       0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                                                       1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 
                                                       1.7, 1.8, 1.9, 2.0
                                                       ])
    # parser.add_argument('--sigma', type=list, default=[0.0, 0.5, 1.0, 1.5, 2.0
    #                                                    ])
    parser.add_argument('--adv', type=str, default="Adv")


    parser.add_argument('--num_steps', type=int, default=10)
    parser.add_argument('--num-noise-vec', default=1, type=int,
                    help="number of noise vectors to use for finding adversarial examples. `m_train` in the paper.")
    parser.add_argument('--no-grad-attack', action='store_true',
                    help="Choice of whether to use gradients during attack or do the cheap trick")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    for sigma in args.sigma:
        # args.save_folder = "logs/" + args.adv + f"/{sigma}/"
        args.save_folder = "logs/" + "NewLoss" + f"/{sigma}/"

        torch.manual_seed(args.seed)
        if not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)

        train_loader, test_loader = data_preprocess.load(
            args.data_folder, batch_size=args.batchsize)
        
        model = net.Network().to(DEVICE)

        optimizer = optim.SGD(params=model.parameters(), lr=args.lr, momentum=args.momentum)

        result = train(model, optimizer, train_loader, test_loader, args, sigma)
        torch.save(model, args.save_folder + "weight.pt")

        result = np.array(result, dtype=float)
        np.savetxt(args.save_folder + "result.csv", result, fmt='%.2f', delimiter=',')
        plot(args)
