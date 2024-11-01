import os
from PIL import ImageEnhance, Image, ImageChops
from tqdm import tqdm

from datasets import get_dataset

ds = get_dataset("causalface_smiling", remove_background=False)

df = ds.meta_df()
df = df[df.smiling == 0.0]

brightness_levels = [0.92, 0.95, 0.98, 1.0, 1.02, 1.03, 1.05, 1.07, 1.10, 1.13]

out_dir = "./data/causalface/final_picked_brightness"
os.makedirs(out_dir, exist_ok=True)

for img_path in tqdm(df.img_path):
    img = ds.pil_load(img_path)
    enhancer = ImageEnhance.Brightness(img)
    for level in brightness_levels:
        img2 = enhancer.enhance(level)
        new_image_path = (img_path
                          .replace("final_picked_smiling", "final_picked_brightness")
                          .replace("smiling_0_o2", f"brightness_{level}")
                          )
        os.makedirs(os.path.dirname(new_image_path), exist_ok=True)
        #print(img_path)
        #print(new_image_path)
        img2.save(new_image_path)
