from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

def func(t, v, k):
    """ computes the function S(t) with constants v and k """
    
    # TODO: return the given function S(t)
    return v * (t - ((1 - np.exp(-1*k*t)))/k)
    # END TODO


def find_constants(df: pd.DataFrame, func: Callable):
    """ returns the constants v and k """

    v = 0
    k = 0

    # TODO: fit a curve using SciPy to estimate v and k
    p_opt, p_cov = curve_fit(func, df['t'], df['S'])
    v, k = p_opt
    # END TODO

    return v, k


if __name__ == "__main__":
    df = pd.read_csv("data.csv")
    v, k = find_constants(df, func)
    v = v.round(4)
    k = k.round(4)
    print(v, k)

    # TODO: plot a histogram and save to fit_curve.png
    plt.scatter(df['t'], df['S'], label='data', marker="*",color='blue')
    plt.plot(df['t'],func(df['t'], v, k),label="v = " + str(v) + ", k = " + str(k) ,color="red")
    plt.xlabel("t")
    plt.ylabel("S")
    plt.legend()
    plt.savefig("fit_curve.png")
    # END TODO
