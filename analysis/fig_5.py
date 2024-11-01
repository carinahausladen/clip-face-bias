"extension to intersectionality lineplots; i keep both until we have final version"
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import image as mpimg

from attribute_models import ABCModel, StereotypeContentModel

DELTA = True
datasets = ['causalface_age', 'fairface', 'utkface']


def main(model: str, add_example_photos: bool = False):
    # Preprocessing functions for each dataset
    def fairface_prep(df_fairface):

        df_fairface['race'] = df_fairface['race'].apply(lambda x: 'Asian' if 'Asian' in x else x)
        df_fairface['race'] = df_fairface['race'].str.lower()
        df_fairface = df_fairface[df_fairface['race'].isin(['black', 'white', 'asian'])]
        df_fairface['gender'] = df_fairface['gender'].str.lower()

        def age_range_to_midpoint(age_range):
            if age_range == 'more than 70':
                return 70
            else:
                lower, upper = age_range.split('-')
                return (int(lower) + int(upper)) / 2

        df_fairface['age'] = df_fairface['age'].apply(age_range_to_midpoint)
        df_fairface = df_fairface[df_fairface['age'] >= 20]
        return df_fairface

    def utkface_prep(df_utkface):
        # Filter out ages below 20
        df_utkface = df_utkface[df_utkface['age'] >= 20]

        # Normalize race and gender values
        df_utkface['race'] = df_utkface['race'].str.lower()
        df_utkface['gender'] = df_utkface['gender'].str.lower()

        # Count the number of samples per age
        age_counts = df_utkface['age'].value_counts()

        # Filter out ages with less than 100 images
        valid_ages = age_counts[age_counts >= 100].index
        df_utkface = df_utkface[df_utkface['age'].isin(valid_ages)]
        min_samples = df_utkface['age'].value_counts().min()
        sampled_dfs = [df_group.sample(min_samples, replace=False, random_state=42)
                       for age, df_group in df_utkface.groupby('age')]
        df_utkface_uniform = pd.concat(sampled_dfs)

        return df_utkface_uniform

    f = 1.2
    fig, axes = plt.subplots(3, 2, figsize=(f * 4.8, f * 6.25), dpi=300)
    # Process each dataset
    for dataset_i, dataset in enumerate(datasets):
        df_in_scm = pd.read_csv(f'./results/{model}/{dataset}_cossim_scm_t2.csv')
        df_in_abc = pd.read_csv(f'./results/{model}/{dataset}_cossim_abc_t2.csv')

        # Preprocess data
        if dataset == 'fairface':
            non_cossim_cols = ['img_path', 'race', 'gender', 'age']
            df_in_scm = fairface_prep(df_in_scm)
            df_in_scm = df_in_scm.set_index(non_cossim_cols)
            df_in_abc = fairface_prep(df_in_abc)
            df_in_abc = df_in_abc.set_index(non_cossim_cols)
        elif dataset == 'utkface':
            non_cossim_cols = ['img_path', 'race', 'gender', 'age']
            df_in_scm = utkface_prep(df_in_scm)
            df_in_scm = df_in_scm.set_index(non_cossim_cols)
            df_in_abc = utkface_prep(df_in_abc)
            df_in_abc = df_in_abc.set_index(non_cossim_cols)
        else:  # causalface
            non_cossim_cols = ['img_id', 'img_path', 'seed', 'demo', 'race', 'gender', 'age']
            df_in_scm = pd.read_csv(f'./results/{model}/causalface_age_cossim_scm_t2.csv').set_index(non_cossim_cols)
            df_in_abc = pd.read_csv(f'./results/{model}/causalface_age_cossim_abc_t2.csv').set_index(non_cossim_cols)

        # Combine SCM and ABC data
        df_in = pd.concat([df_in_scm, df_in_abc], axis=1).reset_index()
        df_in = df_in.loc[:, ~df_in.columns.duplicated()]  # drop duplicate columns

        # --------------- differences
        blank = pd.read_csv(f'./results/{model}/{dataset}_cossim_control_t2.csv').set_index("img_path")[
            "cossim_<blank>"]

        if DELTA:
            df_in.set_index("img_path", inplace=True)
            cossim_cols = [col for col in df_in.columns if col.startswith("cossim_")]
            for c in cossim_cols:
                df_in[c] = (df_in[c] - blank) * 100  # to percentage
            df_in.reset_index(inplace=True)

        # --------------- preprocess attributes
        def preprocess_attributes(model, attribute_type):
            attribute_list = getattr(model, attribute_type)
            attributes = ["cossim_" + item for item in attribute_list]
            return df_in.groupby(['img_path', 'age', 'race', 'gender'])[attributes].transform('mean').mean(axis=1)

        df_in["warmth"] = preprocess_attributes(StereotypeContentModel, "warm")
        df_in["competence"] = preprocess_attributes(StereotypeContentModel, "comp")
        df_in['agency_neg'] = preprocess_attributes(ABCModel, 'agency_neg')
        df_in['agency_pos'] = preprocess_attributes(ABCModel, 'agency_pos')
        df_in['belief_neg'] = preprocess_attributes(ABCModel, 'belief_neg')
        df_in['belief_pos'] = preprocess_attributes(ABCModel, 'belief_pos')
        df_in['communion_neg'] = preprocess_attributes(ABCModel, 'communion_neg')
        df_in['communion_pos'] = preprocess_attributes(ABCModel, 'communion_pos')
        df_mean = df_in[
            ['age', 'race', 'gender', 'warmth', 'competence', 'agency_neg', 'agency_pos', 'belief_neg', 'belief_pos',
             'communion_neg', 'communion_pos']].drop_duplicates()
        df_mean.dropna(inplace=True)

        # --------------- plot clusters
        highlight_color = {
            'female': '#D81B60',
            'male': '#1E88E5'
        }

        def plot_dimension(ax, x, y, title, df_mean, degree=2, cluster_colors=None):
            meanmean = df_mean.groupby(['gender', 'race', 'age']).mean().reset_index()

            if dataset == 'utkface':
                alpha = 0.2
                size = 20
            else:
                alpha = 0.8
                size = 50
            sns.scatterplot(data=meanmean, x=x, y=y, hue="gender", style="race",
                            palette=highlight_color, s=size, ax=ax, alpha=alpha)

            for (gender, race), group_data in df_mean.groupby(['gender', 'race']):
                # Cubic polynomial
                poly_coeffs = np.polyfit(group_data[x], group_data[y], degree)
                poly_fun = np.poly1d(poly_coeffs)
                x_values = np.linspace(group_data[x].min(), group_data[x].max(), 100)
                y_values = poly_fun(x_values)

                # Determine the color based on the cluster
                group_color = cluster_colors.get((gender, race), '#D3D3D3')  # Default color
                if dataset == "utkface":
                    if group_color != '#D3D3D3':
                        zorder = 99  # Put the highlighted clusters in front
                    else:
                        zorder = 98
                else:
                    if group_color != '#D3D3D3':
                        zorder = -2  # Put the highlighted clusters in front
                    else:
                        zorder = -1
                ax.plot(x_values, y_values, color=group_color, linewidth=1.5, zorder=zorder)

            ax.set_title(title, fontsize=10)
            ax.grid(False)
            ax.get_legend().remove()
            ax.set_ylabel(None)
            ax.set_xlabel(None)

            ax.set_xticks([meanmean.age.min(), meanmean.age.max()])
            ax.set_xticklabels(["youngest", "oldest"])

        plot_dimension(axes[dataset_i, 0], 'age', 'warmth', 'Warmth', df_mean,
                       cluster_colors={('female', 'black'): '#D81B60'})
        plot_dimension(axes[dataset_i, 1], 'age', 'competence', 'Competence', df_mean,
                       cluster_colors={('female', 'black'): '#D81B60'})
        # plot_dimension(axes[0, dataset_i], 'age', 'agency_pos', 'Positive Agency', df_mean, cluster_colors={('female', 'black'): '#D81B60'})
        # plot_dimension(axes[1, dataset_i], 'age', 'agency_neg', 'Negative Agency', df_mean, cluster_colors={('female', 'black'): '#D81B60'})
        # plot_dimension(axes[0, 2], 'age', 'belief_pos', 'Progressive Beliefs', df_mean, cluster_colors={('female', 'black'): '#D81B60'})
        # plot_dimension(axes[1, 2], 'age', 'belief_neg', 'Conservative Beliefs', df_mean, cluster_colors={('female', 'black'): '#D81B60'})
        # plot_dimension(axes[0, 3], 'age', 'communion_pos', 'Positive Communion', df_mean, cluster_colors={('female', 'black'): '#D81B60'})
        # plot_dimension(axes[1, 3], 'age', 'communion_neg', 'Negative Communion', df_mean, cluster_colors={('female', 'black'): '#D81B60'})

    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles = handles[1:3] + handles[4:7]
    labels = labels[1:3] + labels[4:7]
    labels = ["women", "men", "Asian", "Black", "White"]  # quickfix
    axes[0, 1].legend(handles, labels, loc='lower left', fontsize="small")
    axes[0, 0].set_ylabel("$\Delta$ Cosine Similarity (%)")
    axes[1, 0].set_ylabel("$\Delta$ Cosine Similarity (%)")
    axes[2, 0].set_ylabel("$\Delta$ Cosine Similarity (%)")

    ## add prefix to every title in upper row
    # axes[0, 0].set_title(f"CausalFace\nWarmth")
    # axes[0, 1].set_title(f"FairFace\nWarmth")
    # axes[0, 2].set_title(f"UTKFace\nWarmth")

    ### INSERT PHOTOS
    for label in axes[0, 0].get_xticklabels():
        label.set_visible(False)

    # remove title on mid and bottom row
    axes[1, 0].set_title("")
    axes[1, 1].set_title("")
    axes[2, 0].set_title("")
    axes[2, 1].set_title("")

    if add_example_photos:
        # IMAGE 1
        img = mpimg.imread(f'data/causalface/final_picked_age/seed_55417/asian_male_age_0.8_o2.png')
        newax = fig.add_axes([0.165, 0.64, 0.05, 0.05], anchor='SW', zorder=1)
        newax.imshow(img)
        newax.axis('off')

        # IMAGE 2
        img = mpimg.imread(f'data/causalface/final_picked_age/seed_55417/asian_male_age_4.4_o2.png')
        newax = fig.add_axes([0.482, 0.64, 0.05, 0.05], anchor='SW', zorder=1)
        newax.imshow(img)
        newax.axis('off')

        # textbox
        fig.text(0.1, 0.64, 'example\nidentity', ha='center', va='bottom', fontsize=8)

        # dataset names as vertical texts
        fig.text(0.025, 0.825, 'CausalFace', ha='center', va='center', fontsize=14, rotation=90)
        fig.text(0.025, 0.5, 'FairFace', ha='center', va='center', fontsize=14, rotation=90)
        fig.text(0.025, 0.175, 'UTKFace', ha='center', va='center', fontsize=14, rotation=90)

    plt.tight_layout()
    plt.subplots_adjust(left=0.175, wspace=0.22, hspace=0.2, bottom=0.05)
    plt.savefig(f"plots/{model}/intersect_age_all3ds_vertical.pdf")

    plt.show()


if __name__ == '__main__':
    main("clip_vit_b_32", add_example_photos=True)
    # set add_example_photos to False if you don't have the image dataset
