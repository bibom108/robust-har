import torch
from scipy.stats import norm, binom_test
import numpy as np
from math import ceil
from statsmodels.stats.proportion import proportion_confint

from itertools import accumulate
import dataset.data_preprocess as data_preprocess
import matplotlib.pyplot as plt
import models.network as net
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import json
import datetime
from time import time

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

                # batch = x.repeat((this_batch_size, 1, 1, 1))
                batch = x.repeat((this_batch_size, 1, 1))
                noise = torch.randn_like(batch, device='cuda') * self.sigma

                predictions = self.base_classifier(batch + noise).argmax(1)
                
                # tmp = self.base_classifier(batch + noise)
                # _, predictions = torch.max(tmp, 0)
                # counts[predictions] += 1

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


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)

    return param


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config.json',
                        help='Json file of settings.')
    parser.add_argument('--data_folder', type=str, default='./')
    parser.add_argument('--seed', type=int, default=69)
    parser.add_argument('--N', type=int, default=1000)
    parser.add_argument('--alpha', type=int, default=0.001)

    return parser


if __name__ == '__main__':
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args) 
    args.update(param) 

    torch.manual_seed(args['seed'])
    train_loader, test_loader = data_preprocess.load("", batch_size=1)

    args['save_folder'] = "predict/" + f"{args['prefix']}/"
    if not os.path.exists(args['save_folder']):
        os.makedirs(args['save_folder'])

    test_sigmas = [0.5]

    args['sigma'] = [0.25]
    args['epsilon'] = [0.5]

    for test_sigma in test_sigmas:
        for sigma in args['sigma']:
            for epsilon in args['epsilon']:
                
                log_file = args['save_folder'] + f"{test_sigma}_{sigma}_{epsilon}.txt"
                f = open(log_file, 'w')
                print("idx\tlabel\tpredict\tcorrect\ttime", file=f, flush=True)
                
                weight_file = "logs/" + f"{args['prefix']}/" + f"{sigma}/" + f"{epsilon}/weight.pt"
                model = torch.load(weight_file)
                model.to(DEVICE)

                smoothed_classifier = Smooth(model, 9, test_sigma)

                for i, (sample, target) in enumerate(test_loader):
                    sample, target = sample.to(DEVICE).float(), target.to(DEVICE).long()

                    before_time = time()
                    prediction = smoothed_classifier.predict(sample, args['N'], args['alpha'], 1)
                    after_time = time()

                    correct = int(prediction == target)

                    print(correct)

                    time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))

                    print("{}\t{}\t{}\t{}\t{}".format(i, target.item(), prediction, correct, time_elapsed), file=f, flush=True)

    f.close()
