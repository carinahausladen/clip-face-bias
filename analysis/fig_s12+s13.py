"""
Code for brightness confound analysis. Appendix F in arxiv paper.
"""

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from datasets import get_dataset

causalface = get_dataset("causalface")


def rgb_to_grayscale(rgb_image):
    return np.dot(rgb_image[..., :3], [0.2989, 0.5870, 0.1140])


def gender_diff_heatmap(as_fraction=False, test=False):
    fig, axs = plt.subplots(2, 2, figsize=(8, 7), layout="constrained")
    axs = axs.flatten()

    df = causalface.meta_df().copy()
    males = df[df.gender == "male"]
    females = df[df.gender == "female"]
    n_pairs = len(males)

    def get_image_pair(idx: int):
        male_sample = males.iloc[idx]
        female_sample = females[(females.race == male_sample.race) &
                                (females.seed == male_sample.seed) &
                                (females.age == male_sample.age)]
        assert len(female_sample) == 1
        female_sample = female_sample.iloc[0]
        race = male_sample.race
        return causalface.get_image_by_path(male_sample.img_path), causalface.get_image_by_path(
            female_sample.img_path), race

    diffs = {"asian": list(), "black": list(), "white": list(), "all": list()}
    for i in tqdm(range(n_pairs)):
        m, f, race = get_image_pair(i)
        if as_fraction:
            m_greater_f = rgb_to_grayscale(np.array(m)) > rgb_to_grayscale(np.array(f))
            f_greater_m = rgb_to_grayscale(np.array(m)) < rgb_to_grayscale(np.array(f))
            diff = m_greater_f.astype(int) - f_greater_m.astype(int)  # important to do it this way. this allows zeros!
        else:
            diff = rgb_to_grayscale(np.array(m)) - rgb_to_grayscale(np.array(f))
        diffs["all"].append(diff)
        diffs[race].append(diff)
        if test and i == 5:
            break

    for i, subs in enumerate(diffs.keys()):
        if len(diffs[subs]) == 0 and test:
            all_diffs = np.stack(diffs["all"])
        else:
            all_diffs = np.stack(diffs[subs])
        n_total = len(all_diffs)
        if as_fraction:
            sum_diffs = np.sum(all_diffs, axis=0)
            frac_diffs = sum_diffs / n_total
            _im = axs[i].imshow(frac_diffs, cmap="bwr", vmin=-1.0, vmax=1.0)
        else:
            mean_diffs = np.mean(all_diffs, axis=0)
            absmax = np.max(np.abs(mean_diffs))
            _im = axs[i].imshow(mean_diffs, cmap="bwr", vmin=-absmax, vmax=absmax)
        axs[i].set_title(subs)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        fig.colorbar(_im, fraction=0.046, pad=0.04, ax=axs[i])

    if as_fraction:
        # fig.suptitle(f"Fraction where grayscale value in pixel is larger for men (positive) or women (negative).")
        plt.savefig("plots/gender_heatmaps_frac.pdf")
    else:
        # fig.suptitle(f"Mean difference in grayscale value per pixel (male - female)")
        plt.savefig("plots/gender_heatmaps_abs.pdf")
    plt.show()


def race_diff_heatmap(as_fraction=True, test=False):
    fig, axs = plt.subplots(2, 3, figsize=(8, 5), layout="constrained")
    axs = axs.flatten()

    df = causalface.meta_df().copy()
    asians = df[df.race == "asian"]
    blacks = df[df.race == "black"]
    whites = df[df.race == "white"]
    n_triplets = len(asians)

    def get_image_triplet(idx: int):
        white_sample = whites.iloc[idx]
        black_sample = blacks[(blacks.gender == white_sample.gender) &
                              (blacks.seed == white_sample.seed) &
                              (blacks.age == white_sample.age)]
        asian_sample = asians[(asians.gender == white_sample.gender) &
                              (asians.seed == white_sample.seed) &
                              (asians.age == white_sample.age)]
        assert len(black_sample) == 1
        assert len(asian_sample) == 1
        black_sample = black_sample.iloc[0]
        asian_sample = asian_sample.iloc[0]
        gender = white_sample.gender
        return causalface.get_image_by_path(asian_sample.img_path), causalface.get_image_by_path(black_sample.img_path), \
            causalface.get_image_by_path(white_sample.img_path), gender

    diffs = {"asian-black-female": list(), "asian-white-female": list(), "black-white-female": list(),
             "asian-black-male": list(), "asian-white-male": list(), "black-white-male": list(),
             }
    for i in tqdm(range(n_triplets)):
        a, b, w, gender = get_image_triplet(i)
        if as_fraction:
            a_greater_b = rgb_to_grayscale(np.array(a)) > rgb_to_grayscale(np.array(b))
            b_greater_a = rgb_to_grayscale(np.array(b)) > rgb_to_grayscale(np.array(a))
            a_b_diff = a_greater_b.astype(int) - b_greater_a.astype(int)
            a_greater_w = rgb_to_grayscale(np.array(a)) > rgb_to_grayscale(np.array(w))
            w_greater_a = rgb_to_grayscale(np.array(w)) > rgb_to_grayscale(np.array(a))
            a_w_diff = a_greater_w.astype(int) - w_greater_a.astype(int)
            b_greater_w = rgb_to_grayscale(np.array(b)) > rgb_to_grayscale(np.array(w))
            w_greater_b = rgb_to_grayscale(np.array(w)) > rgb_to_grayscale(np.array(b))
            b_w_diff = b_greater_w.astype(int) - w_greater_b.astype(int)
        else:
            a_b_diff = rgb_to_grayscale(np.array(a)) - rgb_to_grayscale(np.array(b))
            a_w_diff = rgb_to_grayscale(np.array(a)) - rgb_to_grayscale(np.array(w))
            b_w_diff = rgb_to_grayscale(np.array(b)) - rgb_to_grayscale(np.array(w))

        diffs[f"asian-black-{gender}"].append(a_b_diff)
        diffs[f"asian-white-{gender}"].append(a_w_diff)
        diffs[f"black-white-{gender}"].append(b_w_diff)

        if test and i == 100:
            break

    for i, subs in enumerate(diffs.keys()):
        all_diffs = np.stack(diffs[subs])
        n_total = len(all_diffs)
        if as_fraction:
            sum_diffs = np.sum(all_diffs, axis=0)
            frac_diffs = sum_diffs / n_total
            _im = axs[i].imshow(frac_diffs, cmap="bwr", vmin=-1.0, vmax=1.0)

        else:
            mean_diffs = np.mean(all_diffs, axis=0)
            absmax = np.max(np.abs(mean_diffs))
            _im = axs[i].imshow(mean_diffs, cmap="bwr", vmin=-absmax, vmax=absmax)
        axs[i].set_title(subs.replace("-male", " (male)").replace("-female", " (female)"))
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        fig.colorbar(_im, fraction=0.046, pad=0.04, ax=axs[i])

    if as_fraction:
        # fig.suptitle(f"Fraction where grayscale value in pixel is larger for men (positive) or women (negative).")
        plt.savefig("plots/race_heatmaps_frac.pdf")
    else:
        # fig.suptitle(f"Mean difference in grayscale value per pixel (by race)")
        plt.savefig("plots/race_heatmaps_abs.pdf")
    plt.show()


if __name__ == '__main__':
    race_diff_heatmap(as_fraction=True, test=False)
    race_diff_heatmap(as_fraction=False, test=False)
    gender_diff_heatmap(as_fraction=False, test=False)
    gender_diff_heatmap(as_fraction=True, test=False)
