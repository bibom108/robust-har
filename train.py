from itertools import accumulate
import dataset.data_preprocess as data_preprocess
import matplotlib.pyplot as plt
import models.network
import models.resnet
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
from attacks.attack import Attacker, PGD_L2, DDN
import json
from torch.optim.lr_scheduler import StepLR
import logging, sys


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def requires_grad_(model:torch.nn.Module, requires_grad:bool) -> None:
    for param in model.parameters():
        param.requires_grad_(requires_grad)


def valid(model, test_loader):
    model.eval()
    with torch.no_grad():
        correct, total = 0, 0
        for sample, target in test_loader:
            sample, target = sample.to(DEVICE).float(), target.to(DEVICE).long()

            output = model(sample)
            _, predicted = torch.max(output.data, 1)

            total += target.size(0)
            correct += (predicted == target).sum()
    acc_test = float(correct) * 100 / total
    return acc_test


def plot(data, args):
    plt.figure()
    plt.plot(range(1, len(data[:, 0]) + 1),
             data[:, 0], color='blue', label='train')
    plt.plot(range(1, len(data[:, 1]) + 1),
             data[:, 1], color='red', label='test')
    plt.legend()
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.title('Train and Test Accuracy', fontsize=16)
    plt.savefig(args['save_folder'] + "train_plot.png")


def _train(scheduler, model, optimizer, train_loader, test_loader, attacker, criterion, sigma, epsilon, args):
    result = []
    for e in range(args['nepoch']):
        scheduler.step()

        attacker.max_norm = np.min([epsilon, (e + 1) * epsilon/args['warmup']])
        attacker.init_norm = np.min([epsilon, (e + 1) * epsilon/args['warmup']])

        model.train()
        correct, total_loss = 0, 0
        total = 0
        for sample, target in train_loader:
            sample, target = sample.to(DEVICE).float(), target.to(DEVICE).long()

            noise = torch.randn_like(sample, device='cuda') * sigma
            
            sample_w_noise = sample
            if args['adv']:
                requires_grad_(model, False)
                model.eval()
                sample_w_noise = attacker.attack(model, sample, target, 
                                        noise=noise, 
                                        num_noise_vectors=args['num_noise_vec'], 
                                        no_grad=args['no_grad_attack']
                                        )
                model.train()
                requires_grad_(model, True)
            
            if args['prefix'] != "clean_training":
                sample_w_noise = sample_w_noise + noise
                sample_w_noise = torch.clamp(sample_w_noise, min = 0, max = 1)
            
            output_w_noise = model(sample_w_noise)
            loss = criterion(output_w_noise, target)

            if args['mix']:
                output = model(sample)
                loss_wo_noise = criterion(output, target)
                loss = loss + loss_wo_noise

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(output_w_noise.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum()
        acc_train = float(correct) * 100.0 / len(train_loader.dataset)

        # Testing
        acc_test = valid(model, test_loader)
        logging.info(f"Epoch: [{e}/{args['nepoch']}], LR: {scheduler.get_last_lr()[0]}, "
              f"loss:{total_loss / len(train_loader):.4f}, train_acc: {acc_train:.2f}, "
              f"test_acc: {acc_test:.2f}")
        result.append([acc_train, acc_test])
        # result_np = np.array(result, dtype=float)
        # np.savetxt('logs/result_np.csv', result_np, fmt='%.2f', delimiter=',')
    return result


def train(args):
    train_loader, test_loader = data_preprocess.load("", batch_size=args['batchsize'])
    
    for sigma in args['sigma']:
        for epsilon in args['epsilon']:

            args['save_folder'] = "logs/" + f"{args['prefix']}/" + f"{sigma}/" + f"{epsilon}/" 
            if not os.path.exists(args['save_folder']):
                os.makedirs(args['save_folder'])

            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s => %(message)s',
                handlers=[
                    logging.FileHandler(filename=args['save_folder'] + 'info.log'),
                    logging.StreamHandler(sys.stdout)
                ]
            )

            logging.info(f"Sigma: {sigma} / Epsilon: {epsilon}")

            torch.manual_seed(args['seed'])

            model = models.resnet.resnet18(num_classes=6).to(DEVICE)
            # model = models.network.Network().to(DEVICE)
            optimizer = optim.SGD(params=model.parameters(), lr=args['lr'], momentum=args['momentum'])
            attacker = PGD_L2(steps=args['num_steps'], device='cuda', max_norm=epsilon)
            criterion = nn.CrossEntropyLoss()
            scheduler = StepLR(optimizer, step_size=args['lr_step_size'])

            result = _train(scheduler, model, optimizer, train_loader, test_loader, attacker, criterion, sigma, epsilon, args)
            torch.save(model, args['save_folder'] + "weight.pt")

            result = np.array(result, dtype=float)
            # np.savetxt(args['save_folder'] + "result.csv", result, fmt='%.2f', delimiter=',')
            plot(result, args)


def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args) 
    args.update(param) 

    train(args)


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)

    return param


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config.json',
                        help='Json file of settings.')

    return parser


if __name__ == '__main__':
    main()
