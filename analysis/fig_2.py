import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from analysis._causalface_results_loading_utils import load_all_causalface

ATTRIBUTE_COLS = ['warm', 'comp', 'a+', 'a-', 'b+', 'b-', 'c+', 'c-']

np.random.seed(42)
random.seed(42)

MIN_AGE_DISTANCE = 1.1
MIN_SMILING_DISTANCE = 1.1

COLORBLIND_WONG = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
COLOR_BLIND_TWO = ["#005AB5", "#000000"]
COLOR_BLIND_FEDEX = ["#FB2489"] * 3 + ["#BBBBBB"] + ["#FF9300"] * 3

NS = "#896AC1"
S = "#E66100"
CBF_DICT = {
    "lighting": NS,
    "pose": NS,
    "smiling": NS,
    "age": S,
    "gender": S,
    "race": S,
    "seed": "#BBBBBB",
}


def main(model="clip", subtract_blank=True, violin=False):
    assert model in ["clip_vit_b_32", "siglip"]
    df = load_all_causalface(model=model, control=True, scm=True, abc=True, cossim_prefix=None, aggregate=True)

    if subtract_blank:
        for col in ['white', 'black', 'asian', 'male', 'female', 'young', 'old'] + ATTRIBUTE_COLS:
            df[col] = df[col] - df['<blank>']
        delta_str = "delta_"
        ABSDIFF_LABEL = "Abs Diff in $\Delta$ CosSim %"
    else:
        delta_str = ""
        ABSDIFF_LABEL = "Abs Diff in CosSim %"

    def sample(by: str):
        if by == "seed":
            must_match_cols = ["race", "gender", "age", "lighting", "pose", "smiling"]
        elif by == "race":
            must_match_cols = ["seed", "gender", "age", "lighting", "pose", "smiling"]
        elif by == "gender":
            must_match_cols = ["seed", "race", "age", "lighting", "pose", "smiling"]
        else:
            must_match_cols = ["seed", "race", "gender"]

        subset = df.dropna(subset=by)

        first_sample = subset.sample()
        by_value = first_sample[by].values[0]
        if isinstance(by_value, str):
            by_value = f"'{by_value}'"

        must_match_vals = []
        must_null_vals = []
        for col in must_match_cols:
            val = first_sample[col].values[0]
            if isinstance(val, np.float64) and np.isnan(val):
                must_null_vals.append(col)
            else:
                if isinstance(val, str):
                    val = f"'{val}'"
                must_match_vals.append((col, val))
        query_string = ' and '.join([f"{col} == {val}" for col, val in must_match_vals])
        if must_null_vals:
            query_string += " and " + ' and '.join([f"{col}.isnull()" for col in must_null_vals])
        query_string += f" and {by} != {by_value}"

        matching_subset = subset.query(query_string)
        if by == "age":
            matching_subset = matching_subset[abs(matching_subset.age - first_sample.age.values[0]) > MIN_AGE_DISTANCE]
        elif by == "smiling":
            matching_subset = matching_subset[
                abs(matching_subset.smiling - first_sample.smiling.values[0]) > MIN_SMILING_DISTANCE]

        second_sample = matching_subset.sample()
        out = {
            colname: abs(first_sample[colname].values[0] - second_sample[colname].values[0])
            for colname in ATTRIBUTE_COLS
        }
        out["changed_dim"] = by
        return out

    n_samples_per_dim = 1000
    results = pd.DataFrame(

        [sample("lighting") for _ in tqdm(range(n_samples_per_dim))] +
        [sample("pose") for _ in tqdm(range(n_samples_per_dim))] +
        [sample("smiling") for _ in tqdm(range(n_samples_per_dim))] +
        [sample("seed") for _ in tqdm(range(n_samples_per_dim))] +
        [sample("age") for _ in tqdm(range(n_samples_per_dim))] +
        [sample("gender") for _ in tqdm(range(n_samples_per_dim))] +
        [sample("race") for _ in tqdm(range(n_samples_per_dim))]
    )
    sns.set(font_scale=1.5)
    sns.set_style("white")
    plt.rcParams['xtick.bottom'] = True
    plt.rcParams['ytick.left'] = True

    def assign_valence(x):
        if x.endswith("-"):
            return "negative"
        elif x.endswith("+") or x in ["warm", "comp"]:
            return "positive"
        return np.nan

    sns.color_palette(COLORBLIND_WONG)
    results_long = results.melt(id_vars="changed_dim", value_name=ABSDIFF_LABEL, var_name="attribute")
    results_long["valence"] = results_long["attribute"].apply(lambda x: assign_valence(x))

    df_pos_and_neg = results_long[results_long["valence"].isin(["positive", "negative"])]

    def plot_boxplot(data_, ax, ylabel_off=False, title=None, rotation=30, violin=False):
        data = data_.copy()
        data[ABSDIFF_LABEL] *= 100  # to percentage
        order = data.groupby("changed_dim")[ABSDIFF_LABEL].median().sort_values().index

        if violin:
            violin_upper_clip = 100.
            sns.violinplot(data=data[data[ABSDIFF_LABEL] <= violin_upper_clip], ax=ax, order=order, cut=0,
                           x="changed_dim", y=ABSDIFF_LABEL, palette=[CBF_DICT[a] for a in order])
        else:
            sns.boxplot(data=data, ax=ax, showmeans=True,
                        meanprops={"markerfacecolor": "black", "markeredgecolor": "black"}, order=order,
                        x="changed_dim", y=ABSDIFF_LABEL, palette=[CBF_DICT[a] for a in order],
                        showfliers=False)

        ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation, ha="right", rotation_mode='anchor')
        ax.set(xlabel=None)
        # Adjust alpha of box colors
        for patch in ax.artists:
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, .5))  # Set alpha to 0.5
        if ylabel_off:
            ax.set(ylabel=None)
        if title:
            ax.set_title(title)

    _scale_factor = 0.75
    fig, ax = plt.subplots(figsize=(8 * _scale_factor, 6 * _scale_factor), constrained_layout=True)
    plot_boxplot(df_pos_and_neg, ax, rotation=45, violin=violin)
    plt.savefig(f"plots/{model}/{delta_str}combined_absdiff_valence{'_violin' if violin else ''}.pdf")
    plt.show()

    # For statistical test:
    df = results_long.copy()
    # -------------------- Table B3: t-test for all possible pairs
    df_pos = df[df["valence"] == "positive"]
    df_neg = df[df["valence"] == "negative"]

    def compute_ttest(df):
        categories = df['changed_dim'].unique()
        results = []
        for i in range(len(categories)):
            for j in range(len(categories)):
                if not i < j:
                    continue
                cat1 = categories[i]
                cat2 = categories[j]

                data1 = df[df["changed_dim"] == cat1][ABSDIFF_LABEL]
                data2 = df[df["changed_dim"] == cat2][ABSDIFF_LABEL]

                # test normality
                from scipy.stats import mannwhitneyu

                t_stat, p_value = mannwhitneyu(data1, data2, alternative='two-sided')

                results.append((cat1, cat2, t_stat, p_value))

        df_results = pd.DataFrame(results, columns=['Category1', 'Category2', 'test statistic', 'p-value'])
        df_results = df_results[df_results['p-value'] > 0.001]
        df_results = df_results[df_results['Category2'].isin(['age', 'gender', 'race'])]
        return df_results

    merged_df = pd.concat([df_pos, df_neg], ignore_index=True)
    ttest_result = compute_ttest(merged_df)
    print(ttest_result)


if __name__ == '__main__':
    main(model="clip_vit_b_32", subtract_blank=True, violin=False)
