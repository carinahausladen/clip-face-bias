"""Vectorized NDKL calculation."""

import time

import numpy as np
import pandas as pd

np.random.seed(42)

N = 3300

df = pd.DataFrame()
df['gender'] = np.array(["male"] * N + ["female"] * N)
df['value'] = np.concatenate([np.random.randn(N), np.random.randn(N) + 0.1])


def time_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Time {func.__name__}: {end_time - start_time} seconds.")
        return result

    return wrapper


def KL_divergence(D1, D2):
    return np.sum([D1[key] * np.log(D1[key] / D2[key]) for key in D1 if key in D2 and D1[key] != 0])


@time_decorator
def NDKL_for_column(df, cossim, attribute, Dr):
    df_sorted = df.sort_values(by=cossim, ascending=False)
    NDKL = 0
    Z = np.sum(1 / np.log2(np.arange(1, len(df_sorted) + 1) + 1))
    for i in range(1, len(df_sorted) + 1):
        top_i_attr_counts = df_sorted.head(i)[attribute].value_counts(normalize=True).reindex(
            df_sorted[attribute].unique(), fill_value=0)
        Dtau_i_r = {key: top_i_attr_counts.get(key, 0) for key in Dr}
        NDKL += (1 / np.log2(i + 1)) * KL_divergence(Dtau_i_r, Dr)
    return NDKL / Z


@time_decorator
def NDKL_vectorized(df, cossim, attribute, Dr):
    df_sorted = df.sort_values(by=cossim, ascending=False)
    Z = np.sum(1 / np.log2(np.arange(1, len(df_sorted) + 1) + 1))
    cumsum_counts = pd.get_dummies(df_sorted[attribute]).cumsum()
    cumsum_fracs = cumsum_counts.div(cumsum_counts.sum(axis=1), axis=0)
    for attr_name, attr_val in Dr.items():
        cumsum_fracs[f"desired_{attr_name}"] = attr_val
        cumsum_fracs[f"KLD_{attr_name}"] = (cumsum_fracs[attr_name] * np.log(
            cumsum_fracs[attr_name] / cumsum_fracs[f"desired_{attr_name}"])).fillna(0.0)
    KLD = cumsum_fracs[[f"KLD_{attr_name}" for attr_name in Dr]].sum(axis=1)
    NDKL = (1 / np.log2(np.arange(1, len(KLD) + 1) + 1)) * KLD
    return NDKL.sum() / Z


if __name__ == '__main__':
    desired_dist = {"male": 0.5, "female": 0.5}
    ndkl = NDKL_for_column(df, "value", "gender", desired_dist)
    print(ndkl)

    ndkl = NDKL_vectorized(df, "value", "gender", desired_dist)
    print(ndkl)
