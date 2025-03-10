import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def display_barier(x, y, xlabel, ylabel="barier", label=""):
    nrows = len(x)
    ncols = y.shape[-1]
    de = np.zeros((nrows, ncols - 1))
    for jj in range(ncols - 1):
        de[:, jj] = y[:, jj + 1] - y[:, jj]
    barier = np.max(de, axis=1)
    print(barier)
    # barier = de[:, 2]
    plt.plot(x, barier, marker="o", markerfacecolor="r", label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
