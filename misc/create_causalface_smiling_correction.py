"""Script to correct causalface smiling image by brightness and save corrected images."""

import numpy as np
from tqdm import tqdm
from PIL import Image, ImageEnhance

from datasets.causalface import CausalFaceDataset

SMILING_VALS = [-2.5, -1.5, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4]
RM_BACKGROUND = True


def img_to_grayscale(image):
    grayscale_image = image.convert("L")  # Convert image to grayscale
    np_image = np.array(grayscale_image)  # Convert to numpy array
    return np_image


def calculate_brightness_difference(ref_img, new_img):
    ref_img = img_to_grayscale(ref_img)
    new_img = img_to_grayscale(new_img)

    if RM_BACKGROUND:
        ref_mask = ref_img < 255
        new_mask = new_img < 255

        ref_img = ref_img[ref_mask]
        new_img = new_img[new_mask]

    brightness1 = np.mean(ref_img)
    brightness2 = np.mean(new_img)

    # Calculate percentage difference
    difference = (brightness2 - brightness1) / brightness1
    return difference


def get_image_path(_df, smiling_val):
    return _df.loc[_df.smiling == smiling_val, "img_path"].values[0]


if __name__ == '__main__':
    ds = CausalFaceDataset(subset="smiling", remove_background=False)
    df = ds.meta_df()

    all_out = []
    for seed in tqdm(df.seed.unique()):
        print(seed)
        for demo in df.demo.unique():
            subset = df[(df.seed == seed) & (df.demo == demo)]
            ref_image_path = subset.loc[subset.smiling == 0, "img_path"].values[0]
            ref_image_path_no_bg = ref_image_path.replace(".png", "_rm_bg.png")
            ref_image = Image.open(ref_image_path)
            for img_path, smiling in subset[["img_path", "smiling"]].values:
                img_path_no_bg = img_path.replace(".png", "_rm_bg.png")
                image = Image.open(img_path)
                brightness_diff = calculate_brightness_difference(Image.open(ref_image_path_no_bg),
                                                                  Image.open(img_path_no_bg))
                factor = 1. / (1. + brightness_diff)
                enhancer = ImageEnhance.Brightness(image)
                corrected_image = enhancer.enhance(factor)
                corrected_image.save(img_path.replace(".png", "_corrected.png"))
