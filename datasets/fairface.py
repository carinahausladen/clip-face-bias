import os
#os.chdir("..")  # set working dir to project root

from PIL import Image
import pandas as pd


class FairfaceDataset:

    def __init__(self, basepath: str = "data/fairface"):
        self.basepath = basepath

        self._meta = self.meta_df()

    def meta_df(self):
        df = pd.read_csv(os.path.join(self.basepath, "fairface_label_train.csv"))
        df["photo_id"] = df["file"].apply(lambda x: x.split("/")[-1].split(".")[0]).astype(int)
        df = df.drop(columns=["file", "service_test"])
        df["img_path"] = df["photo_id"].apply(
            lambda x: os.path.join(self.basepath, f"fairface-img-margin025-trainval/train/{x}.jpg"))
        df["age"] = pd.Categorical(df["age"], ordered=True, categories=['0-2', '3-9', '10-19', '20-29', '30-39',
                                                                        '40-49', '50-59', '60-69', 'more than 70'])
        df.set_index("photo_id", inplace=True)
        return df

    def get_image(self, idx: int):
        img_path = os.path.join(self.basepath, f"fairface-img-margin025-trainval/train/{idx}.jpg")
        return self.get_image_by_path(img_path)

    def get_image_by_path(self, img_path: str):
        img = Image.open(img_path)
        return img

    def __len__(self):
        return len(self._meta)

    def __getitem__(self, idx):
        return self.get_image(self._meta.iloc[idx].name)


if __name__ == '__main__':
    """Creates fairface example image"""
    import matplotlib.pyplot as plt

    ds = FairfaceDataset()
    df = ds.meta_df()
    df = df[df["age"] > '10-19']

    n_rows = 8
    n_cols = 15

    whites = df[df["race"] == "White"].sample(n_rows * n_cols // 3).index.values
    blacks = df[df["race"] == "Black"].sample(n_rows * n_cols // 3).index.values
    asians = df[df["race"].str.contains("Asian")].sample(n_rows * n_cols // 3).index.values

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

    plt.savefig("fairface.png")
    plt.show()
