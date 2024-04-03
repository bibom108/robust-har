import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from typing import *
import pandas as pd
import seaborn as sns
import math
import os

sns.set()


class Accuracy(object):
    def at_radii(self, radii: np.ndarray):
        raise NotImplementedError()


class ApproximateAccuracy(Accuracy):
    def __init__(self, data_file_path: str):
        self.data_file_path = data_file_path

    def at_radii(self, radii: np.ndarray) -> np.ndarray:
        df = pd.read_csv(self.data_file_path, delimiter="\t")
        return np.array([self.at_radius(df, radius) for radius in radii])

    def at_radius(self, df: pd.DataFrame, radius: float):
        return (df["correct"] & (df["radius"] >= radius)).mean()


class HighProbAccuracy(Accuracy):
    def __init__(self, data_file_path: str, alpha: float, rho: float):
        self.data_file_path = data_file_path
        self.alpha = alpha
        self.rho = rho

    def at_radii(self, radii: np.ndarray) -> np.ndarray:
        df = pd.read_csv(self.data_file_path, delimiter="\t")
        return np.array([self.at_radius(df, radius) for radius in radii])

    def at_radius(self, df: pd.DataFrame, radius: float):
        mean = (df["correct"] & (df["radius"] >= radius)).mean()
        num_examples = len(df)
        return (mean - self.alpha - math.sqrt(self.alpha * (1 - self.alpha) * math.log(1 / self.rho) / num_examples)
                - math.log(1 / self.rho) / (3 * num_examples))
    

class EmpiricalAccuracy(Accuracy):
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def at_radii(self, radii: np.ndarray, attack: str, method: str) -> np.ndarray:
        accuracies = []
        for radius in radii:
            file_path = os.path.join(self.data_dir, '{}_{:.3f}/{}/predictions'.format(attack, radius, method))
            df = pd.read_csv(file_path, delimiter="\t")
            accuracies.append(self.at_radius(df, radius))
        return np.array(accuracies)

    def at_radius(self, df: pd.DataFrame, radius: float):
        return df["correct"].mean()


class Line(object):
    def __init__(self, quantity: Accuracy, legend: str, plot_fmt: str = "", scale_x: float = 1):
        self.quantity = quantity
        self.legend = legend
        self.plot_fmt = plot_fmt
        self.scale_x = scale_x


def plot_certified_accuracy(outfile: str, title: str, max_radius: float,
                            lines: List[Line], radius_step: float = 0.01) -> None:
    radii = np.arange(0, max_radius + radius_step, radius_step)
    plt.figure()
    for line in lines:
        plt.plot(radii * line.scale_x, line.quantity.at_radii(radii), line.plot_fmt)

    plt.ylim((0, 1))
    plt.xlim((0, max_radius))
    plt.tick_params(labelsize=14)
    plt.xlabel("radius", fontsize=16)
    plt.ylabel("certified accuracy", fontsize=16)
    plt.legend([method.legend for method in lines], loc='upper right', fontsize=16)
    plt.savefig(outfile + ".pdf")
    plt.tight_layout()
    plt.title(title, fontsize=20)
    plt.tight_layout()
    plt.savefig(outfile + ".png", dpi=300)
    plt.close()


def plot_empirical_accuracy_vary_N(
                                    outfile: str, 
                                    title: str,
                                    max_radius: float,
                                    methods_certified: list,
                                    methods_empirical: list,
                                    radius_step: float = 0.125) -> None:

    plt.figure()

    # if methods_certified is not None:
    #     accuracies_cert_cohen, radii = _get_accuracies_at_radii(methods_certified, 0, max_radius, 0.01)
    #     plt.plot(radii, accuracies_cert_cohen.max(0), label='Cohen et al. certified')

    empirical_radii = np.arange(0, max_radius + radius_step, radius_step)
    for method in methods_empirical:
        N = method.split('N')[-1]
        emp_acc = EmpiricalAccuracy(method)
        accuracies_empirical = emp_acc.at_radii(empirical_radii, attack='PGD', method='num_128')
        plt.plot(empirical_radii, accuracies_empirical, dashes=[6, 2], label='n = {}'.format(N))

    plt.ylim((0, 1))
    plt.xlim((0, max_radius))
    plt.tick_params(labelsize=14)
    plt.xlabel("$\ell_2$ radius", fontsize=16)
    plt.ylabel("Accuracy", fontsize=16)
    plt.legend(loc='upper right', fontsize=16)
    plt.tight_layout()
    plt.savefig(outfile + ".pdf")
    plt.title(title, fontsize=20)
    plt.tight_layout()
    plt.savefig(outfile + ".png", dpi=300)
    plt.close()

