import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools

from scipy.integrate import simps

colorblind_palette = sns.color_palette("colorblind", 10)
colorblind_palette[5] = colorblind_palette[7]
sns.set_palette(colorblind_palette)

replace_dict = {
    "asian_female": "Asian Female",
    "asian_male": "Asian Male",
    "black_female": "Black Female",
    "black_male": "Black Male",
    "white_female": "White Female",
    "white_male": "White Male",
}

replace_dict_short = {
    "asian_female": "AF",
    "asian_male": "AM",
    "black_female": "BF",
    "black_male": "BM",
    "white_female": "WF",
    "white_male": "WM",
}


### 1 a
def fig1(model: str, kde=False, box=True, annotate_box=True):
    df = pd.read_csv(f'./results/{model}/causalface_smiling_cossim_control_t2.csv')
    prototypes = df.loc[df.smiling == 0.0, ["seed", "demo", "cossim_<blank>"]]
    prototypes.demo = prototypes.demo.map(replace_dict)

    if kde:
        # sns.histplot(prototypes, x="cossim_<blank>", hue="demo", bins=bins, element="poly", kde=True, fill=False)
        sns.kdeplot(prototypes, x="cossim_<blank>", hue="demo")
        plt.xlabel("Cosine Similarity to Neutral Prompt")
        # add gridlines
        plt.grid(True)
        plt.savefig(f"plots/{model}/neutral_kde.pdf")
        plt.show()

    if box:
        ax = sns.boxplot(data=prototypes, y="cossim_<blank>", x="demo", hue="demo", showfliers=True, showmeans=True,
                         meanprops={'marker': '^', 'markerfacecolor': 'lightgrey', 'markeredgecolor': 'lightgrey'},
                         flierprops=dict(marker='o', markerfacecolor='white', markeredgecolor='black')
                         )
        # rotate x labels
        plt.xticks(rotation=45)
        # make xtick labels so their end (right side) aligns with tick
        plt.xticks(ha='right')

        plt.xlabel(None)
        plt.ylabel("Cosine Similarity to Neutral Prompt")

        if annotate_box:
            from statannotations.Annotator import Annotator
            pairs = list(itertools.combinations(prototypes.demo.unique(), 2))
            annotator = Annotator(ax, pairs, data=prototypes, y="cossim_<blank>", x="demo")
            annotator.configure(test='t-test_paired', text_format='star', loc='inside', hide_non_significant=True)
            annotator.apply_and_annotate()

        plt.tight_layout()
        plt.savefig(f"plots/{model}/boxplot{'_annot' if annotate_box else ''}.pdf")
        plt.show()


###1 b
def fig2(model: str, vertical=True):
    df = pd.read_csv(f'./results/{model}/causalface_smiling_cossim_control_t2.csv').set_index("img_id")
    prototypes = df.loc[df.smiling == 0.0, :].copy()
    prototypes.demo = prototypes.demo.map(replace_dict)

    demo_combs = [_ for _ in itertools.combinations(prototypes.demo.unique(), 2)]
    prototypes.set_index("seed", inplace=True)

    if vertical:
        fig, axs = plt.subplots(5, 3, figsize=(7, 10), constrained_layout=True)
    else:
        fig, axs = plt.subplots(3, 5, figsize=(10, 5.5), constrained_layout=True)

    for ax, (demo1, demo2) in zip(axs.flatten(), demo_combs):
        demo1_prototypes = prototypes.loc[prototypes.demo == demo1, "cossim_<blank>"]
        demo2_prototypes = prototypes.loc[prototypes.demo == demo2, "cossim_<blank>"]
        subset = pd.DataFrame({"demo1": demo1_prototypes, "demo2": demo2_prototypes})
        sns.scatterplot(data=subset, x="demo1", y="demo2", ax=ax, marker="o", alpha=0.5)

        # add mean values as triangles
        ax.scatter(subset["demo1"].mean(), subset["demo2"].mean(), marker="^", alpha=0.5, edgecolor="white",
                   color="red", s=50, zorder=99)

        # set x and x labels
        ax.set_xlabel(demo1)
        ax.set_ylabel(demo2)

        # add correlation coefficient to the bottom right of each panel with small font size

        corr = subset.corr().iloc[0, 1]
        print(subset.corr())
        ax.text(0.98, 0.02, f"r={corr:.2f}", ha='right', va='bottom', transform=ax.transAxes, size=7)

    for ax in axs.flatten():
        ax.set_aspect('equal')  # Set aspect ratio to be equal

        if model == "clip":
            _min = 0.21
            _max = 0.29
            ax.set_xlim((_min, _max))  # Set the same x-axis limits
            ax.set_ylim((_min, _max))  # Set the same y-axis limits
            # fix tick positions
            ax.set_xticks([0.22, 0.25, 0.28])
            ax.set_yticks([0.22, 0.25, 0.28])
            # add diagonal x=y line
            ax.plot([_min, _max], [_min, _max], color='black', linestyle='--', linewidth=1, zorder=-1)
        else:
            pass  # TODO

    plt.savefig(f"plots/{model}/scatters_{'v' if vertical else 'h'}.pdf")
    plt.show()


def fig3(model: str):
    ### 2
    df = pd.read_csv(f'./results/{model}/causalface_smiling_cossim_control_t2.csv').set_index("img_id")
    df2 = pd.read_csv(f'./results/{model}/causalface_smiling_cossim_scm_t2.csv').set_index("img_id")
    df["cossim_competent"] = df2["cossim_competent"]
    df["cossim_warm"] = df2["cossim_warm"]
    df.reset_index(inplace=True)

    prototypes = df.loc[df.smiling == 0.0, :].copy()
    prototypes.demo = prototypes.demo.map(replace_dict_short)

    attributes_of_interest = ["black", "asian", "male", "female", "warm", "competent"]
    for attribute in attributes_of_interest:
        prototypes[f"cossim_{attribute}_corrected"] = prototypes[f"cossim_{attribute}"] - prototypes["cossim_<blank>"]

    fig, ax = plt.subplots(len(attributes_of_interest), 3, figsize=(12, 3 * (len(attributes_of_interest))),
                           constrained_layout=True, sharey="all", sharex=False)

    common_norm = False
    prototypes.rename(columns={"demo": "Group"}, inplace=True)
    for a, attribute in enumerate(attributes_of_interest):
        kde0 = sns.kdeplot(prototypes, x="cossim_<blank>", hue="Group", ax=ax[a][0], common_norm=common_norm)
        kde1 = sns.kdeplot(prototypes, x=f"cossim_{attribute}", hue="Group", ax=ax[a][1], common_norm=common_norm)
        kde2 = sns.kdeplot(prototypes, x=f"cossim_{attribute}_corrected", hue="Group", ax=ax[a][2],
                           common_norm=common_norm)

        for kde in [kde0, kde1, kde2]:
            for line in kde.get_lines():
                x, y = line.get_data()
                area = simps(y, x)
                print(f"Integral of the KDE curve for {line.get_label()}: {area}")

        ax[a][0].set_title("neutral")
        ax[a][1].set_title(f"'{attribute}'")
        ax[a][2].set_title(f"'{attribute}' minus neutral")
        # make title bold
        ax[a][0].title.set_fontweight('bold')
        ax[a][1].title.set_fontweight('bold')
        ax[a][2].title.set_fontweight('bold')

        # set x range
        if model == "clip":
            ax[a][0].set_xlim(0.17, 0.33)
            ax[a][1].set_xlim(0.17, 0.33)
            ax[a][2].set_xlim(-0.08, 0.08)
        else:
            pass  # TODO

        ax[a][0].tick_params(axis='x', which='both', labelbottom=True)  # Make sure x labels are shown
        ax[a][0].xaxis.label.set_visible(True)
        ax[a][0].set_xlabel("Cosine Similarity")
        ax[a][1].tick_params(axis='x', which='both', labelbottom=True)  # Make sure x labels are shown
        ax[a][1].xaxis.label.set_visible(True)
        ax[a][1].set_xlabel("Cosine Similarity")
        ax[a][2].tick_params(axis='x', which='both', labelbottom=True)  # Make sure x labels are shown
        ax[a][2].xaxis.label.set_visible(True)
        ax[a][2].set_xlabel("$\Delta$ Cosine Similarity")

    plt.savefig(f"plots/{model}/kde_corrections.pdf")
    plt.show()


if __name__ == '__main__':
    fig1("clip_vit_b_32", annotate_box=False)
    fig2("clip_vit_b_32", vertical=True)
    fig2("clip_vit_b_32", vertical=False)
    fig3("clip_vit_b_32")
