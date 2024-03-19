import torch
from scipy.stats import norm, binom_test
import numpy as np
from math import ceil
from statsmodels.stats.proportion import proportion_confint

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

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Smooth(object):
    """A smoothed classifier g """

    # to abstain, Smooth returns this int
    ABSTAIN = -1

    def __init__(self, base_classifier: torch.nn.Module, num_classes: int, sigma: float):
        """
        :param base_classifier: maps from [batch x channel x height x width] to [batch x num_classes]
        :param num_classes:
        :param sigma: the noise level hyperparameter
        """
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.sigma = sigma

    def certify(self, x: torch.tensor, n0: int, n: int, alpha: float, batch_size: int) -> (int, float):
        """ Monte Carlo algorithm for certifying that g's prediction around x is constant within some L2 radius.
        With probability at least 1 - alpha, the class returned by this method will equal g(x), and g's prediction will
        robust within a L2 ball of radius R around x.

        :param x: the input [channel x height x width]
        :param n0: the number of Monte Carlo samples to use for selection
        :param n: the number of Monte Carlo samples to use for estimation
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: (predicted class, certified radius)
                 in the case of abstention, the class will be ABSTAIN and the radius 0.
        """
        self.base_classifier.eval()
        # draw samples of f(x+ epsilon)
        counts_selection = self._sample_noise(x, n0, batch_size)
        # use these samples to take a guess at the top class
        cAHat = counts_selection.argmax().item()
        # draw more samples of f(x + epsilon)
        counts_estimation = self._sample_noise(x, n, batch_size)
        # use these samples to estimate a lower bound on pA
        nA = counts_estimation[cAHat].item()
        pABar = self._lower_confidence_bound(nA, n, alpha)
        if pABar < 0.5:
            return Smooth.ABSTAIN, 0.0
        else:
            radius = self.sigma * norm.ppf(pABar)
            return cAHat, radius

    def predict(self, x: torch.tensor, n: int, alpha: float, batch_size: int) -> int:
        """ Monte Carlo algorithm for evaluating the prediction of g at x.  With probability at least 1 - alpha, the
        class returned by this method will equal g(x).

        This function uses the hypothesis test described in https://arxiv.org/abs/1610.03944
        for identifying the top category of a multinomial distribution.

        :param x: the input [channel x height x width]
        :param n: the number of Monte Carlo samples to use
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: the predicted class, or ABSTAIN
        """
        self.base_classifier.eval()
        counts = self._sample_noise(x, n, batch_size)
        top2 = counts.argsort()[::-1][:2]
        count1 = counts[top2[0]]
        count2 = counts[top2[1]]
        if binom_test(count1, count1 + count2, p=0.5) > alpha:
            return Smooth.ABSTAIN
        else:
            return top2[0]

    def _sample_noise(self, x: torch.tensor, num: int, batch_size) -> np.ndarray:
        """ Sample the base classifier's prediction under noisy corruptions of the input x.

        :param x: the input [channel x width x height]
        :param num: number of samples to collect
        :param batch_size:
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """
        with torch.no_grad():
            counts = np.zeros(self.num_classes, dtype=int)
            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                batch = x.repeat((this_batch_size, 1, 1, 1))
                noise = torch.randn_like(batch, device='cuda') * self.sigma

                predictions = self.base_classifier(batch + noise).argmax(1)
                counts += self._count_arr(predictions.cpu().numpy(), self.num_classes)
            return counts

    def _count_arr(self, arr: np.ndarray, length: int) -> np.ndarray:
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts

    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.

        This function uses the Clopper-Pearson method.

        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, default='./')
    parser.add_argument('--batchsize', type=int, default=1)
    parser.add_argument('--seed', type=int, default=69)

    parser.add_argument('--sigma_train', type=list, default=[0.0, 1.0,
                                                    2.0
                                                    ])
    parser.add_argument('--adv', type=str, default="Adv")

    parser.add_argument('--sigma', type=int, default=4)
    parser.add_argument('--N0', type=int, default=100)
    parser.add_argument('--N', type=int, default=1000)
    parser.add_argument('--alpha', type=int, default=0.001)
    parser.add_argument('--outfile', type=str, default="certify/mixedloss.txt")

    args = parser.parse_args()
    return args


def plot(data, args):
    plt.figure()

    for i, sigma in enumerate(args.sigma_train):
        plt.plot(list(zip(*data[i]))[0], list(zip(*data[i]))[1], label=sigma)

    for i, sigma in enumerate(args.sigma_train):
        plt.plot(list(zip(*data[len(args.sigma_train) + i]))[0], list(zip(*data[len(args.sigma_train) + i]))[1], label=f"{sigma}_newloss")

    plt.legend()
    plt.xlabel('Sigma', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.title('Sigma and Certified Accuracy', fontsize=16)

    plt.savefig(f"plot_certification.png")


if __name__ == '__main__':
    args = get_args()
    torch.manual_seed(args.seed)
    train_loader, test_loader = data_preprocess.load(args.data_folder, batch_size=args.batchsize)

    data = []
    for sigma_train in args.sigma_train:
        # load_file = "logs/" + args.adv + f"/{sigma_train}/weight.pt"
        load_file = "logs/" + args.adv + f"/{sigma_train}/weight.pt"
        model = torch.load(load_file)
        model.to(DEVICE)
        print(sigma_train)

        sigma = 0
        tmp = []
        while sigma <= args.sigma:
            correct, total = 0, 0
            smoothed_classifier = Smooth(model, 9, sigma)

            for sample, target in test_loader:
                sample, target = sample.to(DEVICE).float(), target.to(DEVICE).long()

                prediction, radius = smoothed_classifier.certify(sample, args.N0, args.N, args.alpha, args.batchsize)

                if prediction == target and radius >= sigma:
                    correct += (prediction == target)
                total += target.size(0)

            sigma += 0.5

            tmp.append([sigma, float(correct) * 100 / total])

        data.append(tmp)


    for sigma_train in args.sigma_train:
        # load_file = "logs/" + args.adv + f"/{sigma_train}/weight.pt"
        load_file = "logs/" + "NewLoss" + f"/{sigma_train}/weight.pt"
        model = torch.load(load_file)
        model.to(DEVICE)
        print(sigma_train)

        sigma = 0
        tmp = []
        while sigma <= args.sigma:
            correct, total = 0, 0
            smoothed_classifier = Smooth(model, 9, sigma)

            for sample, target in test_loader:
                sample, target = sample.to(DEVICE).float(), target.to(DEVICE).long()

                prediction, radius = smoothed_classifier.certify(sample, args.N0, args.N, args.alpha, args.batchsize)

                if prediction == target and radius >= sigma:
                    correct += (prediction == target)
                total += target.size(0)

            sigma += 0.5

            tmp.append([sigma, float(correct) * 100 / total])

        data.append(tmp)
    
    plot(data, args)
        
