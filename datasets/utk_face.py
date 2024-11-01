import os
# os.chdir("..")  # set working dir to project root

from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

import random
import numpy as np


class UTKFaceDataset:
    gender_map = {"0": "male", "1": "female"}
    race_map = {"0": "White", "1": "Black", "2": "Asian", "3": "Indian", "4": "Others"}

    def __init__(self, basepath="./data/UTKFace"):
        self.basepath = basepath

        all_files = [f for f in os.listdir(basepath) if f.endswith(".jpg")]
        race = list()
        gender = list()
        age = list()
        full_filepaths = list()
        for file in all_files:
            _age, _gender, _race, _ = file.split("_")
            _age = int(_age)
            _race = self.race_map[_race]
            _gender = self.gender_map[_gender]
            age.append(_age)
            gender.append(_gender)
            race.append(_race)
            full_filepaths.append(os.path.join(basepath, file))
        df = pd.DataFrame()
        df["age"] = age
        df["gender"] = gender
        df["race"] = race
        df["img_path"] = full_filepaths
        self._meta_df = df

    def __len__(self):
        return len(self._meta_df)

    def __getitem__(self, idx):
        return self.get_image(idx)

    def meta_df(self):
        return self._meta_df

    def get_image(self, idx: int):
        img_path = self._meta_df.iloc[idx]["img_path"]
        return self.get_image_by_path(img_path)

    def get_image_by_path(self, img_path: str):
        img = Image.open(img_path)
        return img


def random_examples():
    """Creates utkface example image"""

    ds = UTKFaceDataset()
    df = ds.meta_df()
    df = df[(df['age'] > 19)]

    n_rows = 8
    n_cols = 15

    whites = df[df["race"] == "White"].sample(n_rows * n_cols // 3).index.values
    blacks = df[df["race"] == "Black"].sample(n_rows * n_cols // 3).index.values
    asians = df[df["race"] == "Asian"].sample(n_rows * n_cols // 3).index.values

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, 12), sharex="all", sharey="all",
                            constrained_layout=True)
    for ax in axs.ravel():
        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        ax.tick_params(axis='both', which='both', labelbottom=False, labelleft=False, left=False, bottom=False)
        for edge, spine in ax.spines.items():
            spine.set_visible(False)

    axs = axs.T.ravel()
    j = 0
    for ids in [whites, blacks, asians]:
        for id_ in ids:
            img = ds.get_image(id_)
            axs[j].imshow(img)
            j += 1

    plt.savefig("utkface.png")
    plt.show()


def sorted_examples():
    ds = UTKFaceDataset()
    df = ds.meta_df()
    df = df[(df['age'] > 19)]

    age_ranges = [
        (20, 29),
        (30, 39),
        (40, 49),
        (50, 59),
        (60, 69),
        (70, 79),
        (80, 89),
    ]

    races = ["Asian", "Black", "White"]

    n_rows = 6
    n_cols = len(age_ranges)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, 9), sharex="all", sharey="all",
                            constrained_layout=True)
    for ax in axs.ravel():
        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        ax.tick_params(axis='both', which='both', labelbottom=False, labelleft=False, left=False, bottom=False)
        for edge, spine in ax.spines.items():
            spine.set_visible(False)

    row_id = 0
    for race in races:
        for gender in ["female", "male"]:
            for col_id, (min_age, max_age) in enumerate(age_ranges):
                id_ = df[(df.age >= min_age) & (df.age <= max_age) & (df.gender == gender) & (df.race == race)].sample(
                    1).index.values[0]
                img = ds.get_image(id_)
                axs[row_id][col_id].imshow(img)
            row_id += 1
    plt.tight_layout()
    plt.savefig(f"utkface_sorted_{seed}.png", dpi=300)
    plt.show()


def map_utk_to_fairface_age(age):
    if age < 3:
        return '0-2'
    elif age < 10:
        return '3-9'
    elif age < 20:
        return '10-19'
    elif age < 30:
        return '20-29'
    elif age < 40:
        return '30-39'
    elif age < 50:
        return '40-49'
    elif age < 60:
        return '50-59'
    elif age < 70:
        return '60-69'
    else:
        return '>70'


if __name__ == '__main__':

    for seed in [0, 41, 42]:
        random.seed(seed)
        np.random.seed(seed)
        sorted_examples()

    random_examples()
