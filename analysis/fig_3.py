import pandas as pd
import matplotlib.image as mpimg

from attribute_models import ABCModel, StereotypeContentModel


def main(model, delta=True, add_example_photos=False):
    non_cossim_cols = ['img_id', 'img_path', 'seed', 'demo', 'race', 'gender', 'smiling']
    df_in_scm = pd.read_csv(f'./results/{model}/causalface_smiling_cossim_scm_t2.csv').set_index(non_cossim_cols)
    df_in_abc = pd.read_csv(f'./results/{model}/causalface_smiling_cossim_abc_t2.csv').set_index(non_cossim_cols)
    df_in = pd.concat([df_in_scm, df_in_abc], axis=1).reset_index()

    df_in = df_in.loc[:, ~df_in.columns.duplicated()]  # drop duplicate columns

    causal_blank = pd.read_csv(f'./results/{model}/causalface_smiling_cossim_control_t2.csv').set_index("img_path")[
        "cossim_<blank>"]

    # --------------- differences
    if delta:
        df_in.set_index("img_path", inplace=True)
        cossim_cols = [col for col in df_in.columns if col.startswith("cossim_")]
        for c in cossim_cols:
            df_in[c] = (df_in[c] - causal_blank) * 100  # to percentage
        df_in.reset_index(inplace=True)

    # --------------- plot ALL
    def preprocess_attributes(model, attribute_type):
        attribute_list = getattr(model, attribute_type)
        attributes = ["cossim_" + item for item in attribute_list]
        return df_in.groupby(['smiling', 'race', 'gender'])[attributes].transform('mean').mean(axis=1)

    df_in["warmth"] = preprocess_attributes(StereotypeContentModel, "warm")
    df_in["competence"] = preprocess_attributes(StereotypeContentModel, "comp")
    df_in['agency_neg'] = preprocess_attributes(ABCModel, 'agency_neg')
    df_in['agency_pos'] = preprocess_attributes(ABCModel, 'agency_pos')
    df_in['belief_neg'] = preprocess_attributes(ABCModel, 'belief_neg')
    df_in['belief_pos'] = preprocess_attributes(ABCModel, 'belief_pos')
    df_in['communion_neg'] = preprocess_attributes(ABCModel, 'communion_neg')
    df_in['communion_pos'] = preprocess_attributes(ABCModel, 'communion_pos')

    df_mean = df_in[
        ['smiling', 'race', 'gender', 'warmth', 'competence', 'agency_neg', 'agency_pos', 'belief_neg', 'belief_pos',
         'communion_neg', 'communion_pos']].drop_duplicates()

    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    highlight_color = {
        'female': '#D81B60',  # Hot pink
        'male': '#1E88E5'  # Blue
    }

    # Plotting

    def plot_dimension(ax, x, y, title, df_mean, degree=2, cluster_colors=None):
        sns.scatterplot(data=df_mean, x=x, y=y, hue="gender", style="race",  # size="age",
                        palette=highlight_color, ax=ax, s=50)
        for (gender, race), group_data in df_mean.groupby(['gender', 'race']):
            # Cubic polynomial
            poly_coeffs = np.polyfit(group_data[x], group_data[y], degree)
            poly_fun = np.poly1d(poly_coeffs)
            x_values = np.linspace(group_data[x].min(), group_data[x].max(), 100)
            y_values = poly_fun(x_values)

            # Determine the color based on the cluster
            group_color = cluster_colors.get((gender, race), '#D3D3D3')  # Default color
            ax.plot(x_values, y_values, color=group_color, linewidth=1.5, zorder=-1)

        ax.set_title(title)
        ax.grid(False)  # Disable grid
        # disable legend
        ax.get_legend().remove()
        ax.set_ylabel(None)
        ax.set_xlabel(None)

        ax.set_xticks([-2.5, 0.0, 4.0])
        ax.set_xticklabels(["most\nfrowning", "neutral", "most\nsmiling"])

    f = 1.2
    fig, axes = plt.subplots(2, 4, figsize=(f * 10, f * 5), sharex="all", dpi=300)  # 2 rows, 3 columns
    plot_dimension(axes[0, 0], 'smiling', 'warmth', 'Warmth', df_mean, cluster_colors={('female', 'black'): '#D81B60'})
    plot_dimension(axes[1, 0], 'smiling', 'competence', 'Competence', df_mean,
                   cluster_colors={('female', 'black'): '#D81B60'})
    plot_dimension(axes[0, 1], 'smiling', 'agency_pos', 'Positive Agency', df_mean,
                   cluster_colors={('female', 'black'): '#D81B60'})
    plot_dimension(axes[1, 1], 'smiling', 'agency_neg', 'Negative Agency', df_mean,
                   cluster_colors={('female', 'black'): '#D81B60'})
    plot_dimension(axes[0, 2], 'smiling', 'belief_pos', 'Progressive Beliefs', df_mean,
                   cluster_colors={('female', 'black'): '#D81B60'})
    plot_dimension(axes[1, 2], 'smiling', 'belief_neg', 'Conservative Beliefs', df_mean,
                   cluster_colors={('female', 'black'): '#D81B60'})
    plot_dimension(axes[0, 3], 'smiling', 'communion_pos', 'Positive Communion', df_mean,
                   cluster_colors={('female', 'black'): '#D81B60'})
    plot_dimension(axes[1, 3], 'smiling', 'communion_neg', 'Negative Communion', df_mean,
                   cluster_colors={('female', 'black'): 'orange', ('female', 'white'): 'orange',
                                   ('female', 'asian'): 'orange'})

    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles = handles[1:3] + handles[4:7]
    labels = labels[1:3] + labels[4:7]
    labels = ["women", "men", "Asian", "Black", "White"]  # quickfix
    axes[1, 0].legend(handles, labels, loc='lower right', fontsize="small")

    axes[0, 0].set_ylabel("$\Delta$ Cosine Similarity (%)")
    axes[1, 0].set_ylabel("$\Delta$ Cosine Similarity (%)")

    ### INSERT PHOTOS
    for label in axes[1, 0].get_xticklabels():
        label.set_visible(False)

    if add_example_photos:
        SEED = 51355
        # IMAGE 1
        img = mpimg.imread(
            f'./data/causalface/final_picked_smiling/seed_{SEED}/asian_female_smiling_-2.5_o2_corrected.png')
        newax = fig.add_axes([0.0575, 0.001, 0.075, 0.075], anchor='SW', zorder=1)
        newax.imshow(img)
        newax.axis('off')

        # IMAGE 2
        img = mpimg.imread(
            f'./data/causalface/final_picked_smiling/seed_{SEED}/asian_female_smiling_0_o2_corrected.png')
        newax = fig.add_axes([0.12, 0.001, 0.075, 0.075], anchor='SW', zorder=1)
        newax.imshow(img)
        newax.axis('off')

        # IMAGE 3
        img = mpimg.imread(
            f'./data/causalface/final_picked_smiling/seed_{SEED}/asian_female_smiling_3_o2_corrected.png')
        newax = fig.add_axes([0.22, 0.001, 0.075, 0.075], anchor='SW', zorder=1)
        newax.imshow(img)
        newax.axis('off')

        # textbox
        fig.text(0.0275, 0.018, 'example\nidentity', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(f"plots/{model}/intersect_smiling.pdf")
    plt.show()


if __name__ == '__main__':
    main("clip_vit_b_32", delta=True, add_example_photos=True)
    # set add_example_photos to False if you don't have the image dataset
