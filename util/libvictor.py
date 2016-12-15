#python 3.5
import pandas as pd
import numpy as np
import functools as ft
import itertools as it
import scipy.stats
import os, re
from collections import defaultdict

# miscellaneous utilities

# STATS
# ================
# convert a pandas series containing indexed data to one representing 
# the PMF and CDF of the data
def get_pmf(series):
    if not type(series) is pd.core.series.Series:
        series = pd.Series(series)
    pmf = series.value_counts().sort_index()
    pmf/=pmf.sum()
    return pmf
def get_cdf(series):
    pmf = get_pmf(series)
    cdf = pmf.cumsum()
    return cdf

# construct a scipy discrete distribution from a pandas Series containing data
# indices will be converted to integers counting from 0
def get_scipy_dist(series):
    pmf = get_pmf(series)
    return scipy.stats.rv_discrete(pmf.iloc[0], pmf.iloc[-1], values=(pmf.index, pmf.values))

# tools for applying the CDF, inverse CDF, and sampling the distribution 
# given a numpy 1-D array or pandas series representing the CDF
def apply_cdf(cdf, x):
    if hasattr(cdf,'index'):
        f = lambda x: cdf.iloc[cdf.index.get_loc(x,'pad')]
        if hasattr(x,'__getitem__'):
            return np.vectorize(f)(x)
        return f(x)
    else:
        return cdf[np.int32(x)]
def apply_invcdf(cdf, x):
    shape = np.shape(x)
    y = cdf.searchsorted(np.ravel(x), side='right')
    if hasattr(cdf,'index'):
        y = cdf.index[y].values
        if not hasattr(x,'__getitem__'):
            y=y[0]
    return np.reshape(y, shape)
def sample_cdf(cdf, n=None):
    return apply_invcdf(cdf, np.random.random(n))


# TEXT
# ================
# return a function which replaces substrings matching re_string with repl_string
def make_re_filter(re_string, repl_string):
    regex = re.compile(re_string)
    return lambda x: re.sub(regex, repl_string, x)


# FUNCTIONAL/GENERIC
# ================
# apply functions in fns iteratively to x
def apply_list(x, fns):
    if len(fns)==0:
        return x
    return apply_list(fns[0](x), fns[1:])
def compose_list(fns):
    return lambda x: apply_list(x, fns)