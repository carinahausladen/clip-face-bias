"""
Create qualitative plots of most/least associated images for a given demogrpahic group.
This is meant to find causalface edge cases.
"""

import os

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from attribute_models import ABCModel, StereotypeContentModel


def adjust_labels(df):
    if 'more than 70' in df['age'].unique():
        df.loc[df['age'] == 'more than 70', 'age'] = '>70'
        df.loc[df['race'] == 'Latino_Hispanic', 'race'] = 'Hispanic'
        df.loc[df['race'] == 'East Asian', 'race'] = 'Asian'
        df.loc[df['race'] == 'Southeast Asian', 'race'] = 'Asian'
        df.loc[df['race'] == 'Middle Eastern', 'race'] = 'Mid East'
        df = df[df['race'].isin(['White', 'Asian', 'Black'])]
        age_order = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '>70']
        df['age'] = pd.Categorical(df['age'], categories=age_order, ordered=True)
        df = df.sort_values('age')

    if "Female" in df["gender"].values:
        df["gender"] = df["gender"].replace("Female", "female")
    if "Male" in df["gender"].values:
        df["gender"] = df["gender"].replace("Male", "male")
    if "Asian" in df["race"].values:
        df["race"] = df["race"].replace("Asian", "asian")
    if "White" in df["race"].values:
        df["race"] = df["race"].replace("White", "white")
    if "Black" in df["race"].values:
        df["race"] = df["race"].replace("Black", "black")

    return df


def preprocess_dataframe_general(df, model, attributes):
    df = adjust_labels(df)
    prefix_cols = lambda atts: ["cossim_" + item for item in atts]

    for attr in attributes:
        df[f'{attr}'] = df[prefix_cols(getattr(model, attr))].mean(axis=1)

    cols_to_retain = ['img_id', 'img_path', 'seed', 'demo', 'race', 'gender', 'age'] + \
                     [f'{attr}' for attr in attributes]
    df = df[cols_to_retain]

    return df


def preprocess_dataframe_abc(df):
    attributes_abc = ['agency_pos', 'agency_neg', 'belief_pos', 'belief_neg', 'communion_pos', 'communion_neg']
    return preprocess_dataframe_general(df, ABCModel, attributes_abc)


def preprocess_dataframe_scm(df):
    attributes_scm = ['warm', 'comp']
    return preprocess_dataframe_general(df, StereotypeContentModel, attributes_scm)


os.makedirs("plots/qualitative", exist_ok=True)

abc_cols = ['agency_pos', 'agency_neg', 'belief_pos', 'belief_neg', 'communion_pos',
            'communion_neg']
scm_cols = ['warm', 'comp']

df = pd.read_csv("results/clip_vit_b_32/causalface_age_cossim_abc_t2.csv")
df_abc = preprocess_dataframe_abc(df)

df = pd.read_csv("results/clip_vit_b_32/causalface_age_cossim_scm_t2.csv")
df_scm = preprocess_dataframe_scm(df)


def plot_top_bottom(dimension, race, gender, age=2.6, n=10):
    assert dimension in abc_cols + scm_cols
    df = df_abc.copy() if dimension in abc_cols else df_scm.copy()
    df = df[(df.race == race) & (df.gender == gender) & (df.age == age)]
    df = df.sort_values(dimension, ascending=False)
    df["img_path"] = df["img_path"].str.replace("/home/", "/Users/")
    top = df.head(n)
    bottom = df.tail(n)

    fig, ax = plt.subplots(2, n, figsize=(20, 5), layout="constrained")
    for i, (_, row) in enumerate(top.iterrows()):
        ax[0, i].imshow(Image.open(row["img_path"]))
        ax[0, i].set_title(f"{row[dimension] * 100:.2f}")
        ax[0, i].tick_params(axis='both', which='both', length=0)
        ax[0, i].set_xticklabels([])
        ax[0, i].set_yticklabels([])
        if i == 0:
            ax[0, i].set_ylabel(f"Top {n}")
    for i, (_, row) in enumerate(bottom.iterrows()):
        ax[1, i].imshow(Image.open(row["img_path"]))
        ax[1, i].set_title(f"{row[dimension] * 100:.2f}")
        ax[1, i].tick_params(axis='both', which='both', length=0)
        ax[1, i].set_xticklabels([])
        ax[1, i].set_yticklabels([])
        if i == 0:
            ax[1, i].set_ylabel(f"Bottom {n}")
    fig.suptitle(f"Dimension: {dimension}\n{race}, {gender}, {age}")
    plt.savefig(f"plots/qualitative/{dim}_{race}_{gender}_{age}.png")
    plt.show()


if __name__ == '__main__':
    for dim in abc_cols + scm_cols:
        for race in ["white", "black", "asian"]:
            for gender in ["male", "female"]:
                plot_top_bottom(dim, race, gender, 2.6, 10)
