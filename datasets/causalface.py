import os
import itertools
from collections import OrderedDict

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
OVERVIEW_OUTPUT_PATH = "plots/overview"

COL_LABELS = ["Asian Female", "Asian Male", "Black Female", "Black Male", "White Female", "White Male"]

CORRECT_SMILING = True  # if True, adjusts brightness for smiling images

class CausalFaceDataset:

    filename_abbr = {
        "age": "age",
        "lighting": "l",
        "pose": "y",
        "smiling": "smiling",
        "brightness": "brightness"
    }

    filename_suffix = {
        "age": "_o2",
        "lighting": "",
        "pose": "",
        "smiling": "_o2_corrected" if CORRECT_SMILING else "_o2",
        "brightness": ""
    }

    races = ["asian", "black", "white"]
    genders = ["female", "male"]
    demographics = [f"{r}_{g}" for r, g in itertools.product(races, genders)]

    available_values = {
        "age": [0.8, 1.4, 2.0, 2.3, 2.6, 2.9, 3.2, 3.5, 3.8, 4.4],
        "lighting": [-1, 0, 1, 2, 3, 4, 5, 6],
        "pose": [-3.0, -1.5, 0.0, 1.5, 3.0],
        "smiling": [-2.5, -1.5, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4],
        "brightness": [0.92, 0.95, 0.98, 1.0, 1.02, 1.03, 1.05, 1.07, 1.10, 1.13]
    }

    crop_vals = (40, 0, 472, 432)  # l, t, r, b
    pose_crop_valus = {
        -3.0: (0, 0, 432, 432),
        -1.5: (0, 0, 432, 432),
        0.0: (40, 0, 472, 432),
        1.5: (80, 0, 512, 432),
        3.0: (80, 0, 512, 432),
    }

    def __init__(self, basepath: str = "./data/causalface", subset= "age", crop=True,
                 remove_background=False):
        assert subset in ["age", "lighting", "pose", "smiling", "brightness"]
        self.subset = subset
        self.crop = crop
        self.remove_background = remove_background
        self.basepath = f"{basepath}/final_picked_{subset}"
        self.available_seeds = [int(filename[5:]) for filename in os.listdir(self.basepath) if filename.startswith("seed_")]

        self.all_img_paths = []
        for seed in self.available_seeds:
            seed_dir = os.path.join(self.basepath, f"seed_{seed}")
            for img in os.listdir(seed_dir):
                if remove_background and "rm_bg" in img:
                    self.all_img_paths.append(os.path.join(seed_dir, img))
                elif not remove_background and "rm_bg" not in img:
                    self.all_img_paths.append(os.path.join(seed_dir, img))

    def meta_df(self):

        rm_bg = ""
        if self.remove_background:
            if self.subset in ["age", "smiling"]:
                rm_bg = "_rm_bg"
            else:
                rm_bg = "_o2_rm_bg"

        cols = ["img_id", "img_path", "seed", "demo", "race", "gender", self.subset]
        data = OrderedDict()
        for col in cols:
            data[col] = []

        img_id = 0
        for seed in self.available_seeds:
            for race in self.races:
                for gender in self.genders:
                    demo = f"{race}_{gender}"
                    for val in self.available_values[self.subset]:
                        img_path = os.path.join(self.basepath, f"seed_{seed}",
                                                f"{race}_{gender}_{self.filename_abbr[self.subset]}_{val}{self.filename_suffix[self.subset]}{rm_bg}.png")
                        if self.subset == "lighting" and self.remove_background:
                            img_path = img_path.replace("_l_", "_lighting_")
                        if not os.path.isfile(img_path):
                            continue
                        data["img_id"].append(img_id)
                        data["img_path"].append(img_path)
                        data["seed"].append(seed)
                        data["demo"].append(demo)
                        data["race"].append(race)
                        data["gender"].append(gender)
                        data[self.subset].append(val)

                        img_id += 1

        return pd.DataFrame(data)

    def pil_load(self, image_path: str):
        return Image.open(image_path)

    def pil_load_cropped(self, image_path: str, crop_vals=None):
        img = Image.open(image_path)
        return img.crop(crop_vals or self.crop_vals)

    def get_image(self, idx: int):
        img_path = self.all_img_paths[idx]
        return self.get_image_by_path(img_path)

    def get_image_by_path(self, img_path: str):
        if self.crop:
            if self.subset == "pose":
                meta = self.meta_df().set_index("img_path")
                pose = meta.loc[img_path].pose
                crop_vals = self.pose_crop_valus[pose]
                return self.pil_load_cropped(img_path, crop_vals)
            else:
                return self.pil_load_cropped(img_path)
        return Image.open(img_path)

    @property
    def n_images(self):
        return len(self.all_img_paths)

    def create_seed_overview_image(self, seed, annotate=True, add_marker=False, show=False):
        n_rows = len(self.demographics)
        n_cols = len(self.available_values[self.subset])

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*2, 12), sharex="all", sharey="all", constrained_layout=True)
        for ax in axs.ravel():
            # Hide the right and top spines
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            ax.tick_params(axis='both', which='both', labelbottom=False, labelleft=False, left=False, bottom=False)
            for edge, spine in ax.spines.items():
                spine.set_visible(False)

        for i, demo in enumerate(self.demographics):
            for j, val in enumerate(self.available_values[self.subset]):
                img_path = os.path.join(self.basepath, f"seed_{seed}",
                                        f"{demo}_{self.filename_abbr[self.subset]}_{val}{self.filename_suffix[self.subset]}.png")
                img = self.get_image_by_path(img_path)
                axs[i, j].imshow(img)

                if add_marker:
                    # add red dot at specific position
                    x_pos = 216
                    y_pos = 144
                    axs[i, j].scatter(x_pos, y_pos, color="red", s=20)

        if annotate:
            plt.suptitle(f"seed={seed}", fontsize=26)

            for ax, label in zip(axs[:, 0], COL_LABELS):
                ax.set_ylabel(label, fontsize=16)

            for ax, label in zip(axs[-1, :], self.available_values[self.subset]):
                ax.set_xlabel(f"{self.subset}={label}", fontsize=16)

        os.makedirs(os.path.join(OVERVIEW_OUTPUT_PATH, self.subset), exist_ok=True)
        plt.savefig(os.path.join(OVERVIEW_OUTPUT_PATH, self.subset, f"seed_{seed}_overview_{self.subset}_{'_corrected' if CORRECT_SMILING else ''}.png"))
        if show:
            plt.show()
        plt.close()

    def create_all_overviews(self):
        for seed in tqdm(self.available_seeds, desc=f"Creating overview plots, subset={self.subset}"):
            self.create_seed_overview_image(seed)


if __name__ == '__main__':
    ds = CausalFaceDataset(subset="smiling")
    #ds.create_seed_overview_image(57460, show=True)
    ds.pil_load_cropped(ds.basepath + "/seed_47180/black_female_smiling_0_o2_corrected.png").show()
    #ds.pil_load_cropped(ds.basepath + "/seed_47180/black_female_smiling_2_o2_corrected.png").show()