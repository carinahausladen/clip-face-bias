import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches

from attribute_models import ABCModel
from datasets.utk_face import map_utk_to_fairface_age
from analysis.util import confidence_ellipse, HandlerEllipse

empty_handle = plt.Line2D([], [], color='none')

NO_AXIS = False
SHARE_AXIS = False
PLOT_ELLIPSES = True

DELTA = "relative"

ELLIPSIS_SIGMA = 2
ELLIPSE_STYLE = dict(
    edgecolor="none",
    linestyle="dashed", facecolor="gray", alpha=0.15
)


def main(model: str):
    df_causal = pd.read_csv(f'results/{model}/causalface_age_cossim_abc_t2.csv')

    df_fairface = pd.read_csv(f'results/{model}/fairface_cossim_abc_t2.csv')
    df_fairface['race'] = df_fairface['race'].apply(lambda x: 'Asian' if 'Asian' in x else x)
    df_fairface['race'] = df_fairface['race'].str.lower()
    df_fairface['gender'] = df_fairface['gender'].str.lower()
    df_fairface.loc[df_fairface['age'] == 'more than 70', 'age'] = '>70'
    df_fairface = df_fairface[df_fairface['age'] >= '20-29']
    age_order = reversed(['20-29', '30-39', '40-49', '50-59', '60-69', '>70'])  # '0-2', '3-9', '10-19',
    df_fairface['age'] = pd.Categorical(df_fairface['age'], categories=age_order, ordered=True)

    df_utkface = pd.read_csv(f'results/{model}/utkface_cossim_abc_t2.csv')
    df_utkface = df_utkface[df_utkface.age >= 20]
    df_utkface['race'] = df_utkface['race'].str.lower()
    df_utkface['gender'] = df_utkface['gender'].str.lower()
    df_utkface['age'] = df_utkface['age'].map(map_utk_to_fairface_age)

    unique_races_causal = df_fairface['race'].unique()
    unique_gender_causal = df_fairface['gender'].unique()
    selected_rows = df_fairface[(df_fairface['race'].isin(unique_races_causal)) &
                                (df_fairface['gender'].isin(unique_gender_causal))]
    df_fairface = selected_rows.copy()
    num_samples_per_group = len(df_fairface) // (
                len(df_fairface['race'].unique()) * len(df_fairface['gender'].unique()))
    samples = []
    for race in df_fairface['race'].unique():
        for gender in df_fairface['gender'].unique():
            group = df_fairface[(df_fairface['race'] == race) & (df_fairface['gender'] == gender)]
            sample = group.sample(min(num_samples_per_group, len(group)), random_state=42)
            samples.append(sample)
    df_fairface = pd.concat(samples).reset_index(drop=True)
    df_fairface = df_fairface[df_fairface['race'].isin(['white', 'asian', 'black'])]

    if DELTA:
        causal_blank = pd.read_csv(f'results/{model}/causalface_age_cossim_control_t2.csv').set_index("img_path")[
            "cossim_<blank>"]
        fairface_blank = pd.read_csv(f'results/{model}/fairface_cossim_control_t2.csv').set_index("img_path")[
            "cossim_<blank>"]
        utkface_blank = pd.read_csv(f'results/{model}/utkface_cossim_control_t2.csv').set_index("img_path")[
            "cossim_<blank>"]

        print("Blank means", causal_blank.mean(), fairface_blank.mean(), utkface_blank.mean())

        for _df, blank in zip([df_causal, df_fairface, df_utkface], [causal_blank, fairface_blank, utkface_blank]):
            _df.set_index("img_path", inplace=True)
            _cossim_cols = [col for col in _df.columns if col.startswith("cossim_")]
            if DELTA is "relative":
                _df[_cossim_cols] = _df[_cossim_cols].subtract(blank, axis=0).div(blank, axis=0)
            else:
                _df[_cossim_cols] = _df[_cossim_cols].subtract(blank, axis=0)
            _df.reset_index(inplace=True)

    all_min = np.inf
    all_max = -np.inf

    mins = {"a": np.inf, "b": np.inf, "c": np.inf}
    maxs = {"a": -np.inf, "b": -np.inf, "c": -np.inf}

    def prefix_columns(attribute_list):
        return ["cossim_" + item for item in attribute_list]

    # ------------------------------ intersectionality
    highlight_color = {
        'female': '#D81B60',  # Hot pink
        'male': '#1E88E5'  # Blue
    }

    def plt_intersect(df_in, ax, x_col, y_col, title, x_mean, y_mean, titles=True, axes_suffix=""):

        df_in[x_col] = df_in[x_col] * 100
        df_in[y_col] = df_in[y_col] * 100

        sns.scatterplot(data=df_in, x=x_col, y=y_col,
                        hue="gender", style="race", size="age",
                        palette=highlight_color, ax=ax,
                        alpha=.5,
                        legend=True,
                        )
        if PLOT_ELLIPSES:
            for _gender, _race in df_in[["gender", "race"]].drop_duplicates().values.tolist():
                subset = df_in[(df_in.gender == _gender) & (df_in.race == _race)]
                confidence_ellipse(subset[x_col], subset[y_col], n_std=ELLIPSIS_SIGMA, ax=ax, **ELLIPSE_STYLE)

        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        allmin = min(xmin, ymin)
        allmax = max(xmax, ymax)
        ax.plot([allmin, allmax], [allmin, allmax], color='black', linestyle='--', linewidth=0.5, zorder=-1,
                label="x=y")
        if titles:
            ax.set_title(title)

        # ensure same range for x and y
        print(allmin, allmax)
        ax.set_xlim(allmin, allmax)
        ax.set_ylim(allmin, allmax)

        if DELTA is "relative":
            cossim_string = "Rel. $\Delta$ Cosine Similarity (%)"
        elif DELTA:
            cossim_string = "$\Delta$ Cosine Similarity (%)"
        else:
            "Cosine Similarity (%)"

        ax.set_xlabel(f"{cossim_string}\nPositive{axes_suffix}")  # x_col
        ax.set_ylabel(f"{cossim_string}\nNegative{axes_suffix}")  # y_col

        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f'{val:.1f}'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f'{val:.1f}'))

    def plt_data(df_in, axes, titles=True):
        # Use attributes from ABCModel class, prefixed
        df_in['mean_agency_pos'] = df_in[prefix_columns(ABCModel.agency_pos)].mean(axis=1)
        df_in['mean_agency_neg'] = df_in[prefix_columns(ABCModel.agency_neg)].mean(axis=1)
        df_in['mean_belief_pos'] = df_in[prefix_columns(ABCModel.belief_pos)].mean(axis=1)
        df_in['mean_belief_neg'] = df_in[prefix_columns(ABCModel.belief_neg)].mean(axis=1)
        df_in['mean_communion_pos'] = df_in[prefix_columns(ABCModel.communion_pos)].mean(axis=1)
        df_in['mean_communion_neg'] = df_in[prefix_columns(ABCModel.communion_neg)].mean(axis=1)

        df = df_in[["age", "gender", "race", "mean_agency_pos", "mean_agency_neg", "mean_belief_pos", "mean_belief_neg",
                    "mean_communion_pos", "mean_communion_neg"]]

        grouped_df = df.groupby(["gender", "race", "age"]).mean()[
            ["mean_agency_pos", "mean_agency_neg", "mean_belief_pos", "mean_belief_neg", "mean_communion_pos",
             "mean_communion_neg"]]
        grouped_df = grouped_df.reset_index()

        means = {
            'agency': (round(grouped_df["mean_agency_pos"].mean(), 2), round(grouped_df["mean_agency_neg"].mean(), 2)),
            'belief': (round(grouped_df["mean_belief_pos"].mean(), 2), round(grouped_df["mean_belief_neg"].mean(), 2)),
            'communion': (
                round(grouped_df["mean_communion_pos"].mean(), 2), round(grouped_df["mean_communion_neg"].mean(), 2))
        }

        plt_intersect(grouped_df, axes[0], "mean_agency_pos", "mean_agency_neg", "Agency", *means['agency'],
                      titles=titles)
        plt_intersect(grouped_df, axes[1], "mean_belief_pos", "mean_belief_neg", "Belief", *means['belief'],
                      titles=titles)
        plt_intersect(grouped_df, axes[2], "mean_communion_pos", "mean_communion_neg", "Communion", *means['communion'],
                      titles=titles)

        all_x_values = grouped_df[["mean_agency_pos", "mean_belief_pos", "mean_communion_pos"]].values.flatten()
        all_y_values = grouped_df[["mean_agency_neg", "mean_belief_neg", "mean_communion_neg"]].values.flatten()
        min_x, max_x = min(all_x_values), max(all_x_values)
        min_y, max_y = min(all_y_values), max(all_y_values)
        global all_min, all_max, mins, maxs
        if SHARE_AXIS == "col":
            mins["a"] = min(mins["a"], grouped_df[["mean_agency_pos"]].values.flatten().min(),
                            grouped_df[["mean_agency_neg"]].values.flatten().min())
            mins["b"] = min(mins["b"], grouped_df[["mean_belief_pos"]].values.flatten().min(),
                            grouped_df[["mean_belief_neg"]].values.flatten().min())
            mins["c"] = min(mins["c"], grouped_df[["mean_communion_pos"]].values.flatten().min(),
                            grouped_df[["mean_communion_neg"]].values.flatten().min())
            maxs["a"] = min(maxs["a"], grouped_df[["mean_agency_pos"]].values.flatten().max(),
                            grouped_df[["mean_agency_neg"]].values.flatten().max())
            maxs["b"] = min(maxs["b"], grouped_df[["mean_belief_pos"]].values.flatten().max(),
                            grouped_df[["mean_belief_neg"]].values.flatten().max())
            maxs["c"] = min(maxs["c"], grouped_df[["mean_communion_pos"]].values.flatten().max(),
                            grouped_df[["mean_communion_neg"]].values.flatten().max())
        elif SHARE_AXIS == "all":
            all_min = min(all_min, min_x, min_y)
            all_max = max(all_max, max_x, max_y)
            for ax in axes:
                ax.plot([all_min, all_max], [all_min, all_max], color='grey', linestyle='--',
                        linewidth=0.5)  # Add this line
        else:
            pass

        for ax in axes:
            ax.legend().remove()
            ax.set_aspect('equal', adjustable='datalim')

    def plt_data_single(df_in, ax, which_dim="agency", titles=True):
        # Use attributes from ABCModel class, prefixed
        df_in['mean_agency_pos'] = df_in[prefix_columns(ABCModel.agency_pos)].mean(axis=1)
        df_in['mean_agency_neg'] = df_in[prefix_columns(ABCModel.agency_neg)].mean(axis=1)
        df_in['mean_belief_pos'] = df_in[prefix_columns(ABCModel.belief_pos)].mean(axis=1)
        df_in['mean_belief_neg'] = df_in[prefix_columns(ABCModel.belief_neg)].mean(axis=1)
        df_in['mean_communion_pos'] = df_in[prefix_columns(ABCModel.communion_pos)].mean(axis=1)
        df_in['mean_communion_neg'] = df_in[prefix_columns(ABCModel.communion_neg)].mean(axis=1)

        df = df_in[["age", "gender", "race", "mean_agency_pos", "mean_agency_neg", "mean_belief_pos", "mean_belief_neg",
                    "mean_communion_pos", "mean_communion_neg"]]

        grouped_df = df.groupby(["gender", "race", "age"]).mean()[
            ["mean_agency_pos", "mean_agency_neg", "mean_belief_pos", "mean_belief_neg", "mean_communion_pos",
             "mean_communion_neg"]]
        grouped_df = grouped_df.reset_index()

        means = {
            'agency': (round(grouped_df["mean_agency_pos"].mean(), 2), round(grouped_df["mean_agency_neg"].mean(), 2)),
            'belief': (round(grouped_df["mean_belief_pos"].mean(), 2), round(grouped_df["mean_belief_neg"].mean(), 2)),
            'communion': (
                round(grouped_df["mean_communion_pos"].mean(), 2), round(grouped_df["mean_communion_neg"].mean(), 2))
        }

        if which_dim == "agency":
            plt_intersect(grouped_df, ax, "mean_agency_pos", "mean_agency_neg", "Agency", *means['agency'],
                          titles=titles, axes_suffix=" " + which_dim.capitalize())
        elif which_dim == "belief":
            plt_intersect(grouped_df, ax, "mean_belief_pos", "mean_belief_neg", "Beliefs", *means['belief'],
                          titles=titles, wich_dim=" " + which_dim.capitalize())
        elif which_dim == "communion":
            plt_intersect(grouped_df, ax, "mean_communion_pos", "mean_communion_neg", "Communion", *means['communion'],
                          titles=titles, axes_suffix=" " + which_dim.capitalize())
        else:
            raise ValueError(f"Unknown dimension {which_dim}")

        ax.legend().remove()

    def full_plot():
        f = 1.25
        fig, axes = plt.subplots(3, 3, sharex=SHARE_AXIS, sharey=SHARE_AXIS, constrained_layout=True,
                                 figsize=(f * 5.25, f * 5.5))
        # fig = plt.figure(constrained_layout=True)

        axes = axes.T
        plt_data(df_causal, axes[0], titles=False)
        plt_data(df_fairface, axes[1], titles=False)
        plt_data(df_utkface, axes[2], titles=False)

        axes[0][0].set_title("CausalFace")
        axes[1][0].set_title("FairFace")
        axes[2][0].set_title("UTKFace")

        if SHARE_AXIS == "col":
            for i in range(2):
                axes[i, 0].plot([mins["a"], maxs["a"]], [mins["a"], maxs["a"]], color='grey', linestyle='--',
                                linewidth=0.5)
                axes[i, 1].plot([mins["b"], maxs["b"]], [mins["b"], maxs["b"]], color='grey', linestyle='--',
                                linewidth=0.5)
                axes[i, 2].plot([mins["c"], maxs["c"]], [mins["c"], maxs["c"]], color='grey', linestyle='--',
                                linewidth=0.5)
        for i, ax in enumerate(axes.flatten()):
            ax.set_title(ax.get_title(), fontsize=11)

            if SHARE_AXIS == "all":
                ax.set_yticks([0.22, 0.24])
                ax.set_yticklabels([22.0, 24.0])
            if NO_AXIS:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_yticklabels([])
                ax.set_xticklabels([])

            if not i in [2, 5, 8]:
                ax.set_xlabel("")
            if not i in [0, 1, 2]:
                ax.set_ylabel("")
            ax.tick_params(axis='x', labelsize=6)  # Adjust x-axis tick label size
            ax.tick_params(axis='y', labelsize=6)  # Adjust y-axis tick label size

        handles, labels = axes[0][0].get_legend_handles_labels()
        print(labels)
        criteria = ["0-2", ">70", "1.2", "4.2", "female", "male", "white", "black", "asian", "x=y", "conf"]
        label_map = {"0-2": "youngest", ">70": "oldest", "1.2": "youngest", "4.2": "oldest"}
        filtered_handles = [handle for handle, label in zip(handles, labels) if label in criteria]
        filtered_labels = [label for label in labels if label in criteria]
        filtered_labels = [label_map.get(l) or l for l in filtered_labels]
        handles, labels = filtered_handles, filtered_labels
        handles.insert(2, empty_handle)
        handles.insert(5, empty_handle)
        import matplotlib.patches as mpatches
        grey_ellipse = mpatches.Ellipse((0, 0), 1, 1, **ELLIPSE_STYLE)
        handles.append(grey_ellipse)
        handles.append(empty_handle)

        labels = ["women", "men", "", "youngest", "oldest", "", "Asian", "Black", "White", "pos=neg",
                  f'Covariance errors ({ELLIPSIS_SIGMA}$\sigma$)', ""]
        for h, l in zip(handles, labels):
            if l in ["youngest", "oldest", "Asian", "Black", "White"]:
                h.set_color('black')
                h.set_alpha(1.0)

        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0), fancybox=True, ncol=4,
                   handler_map={mpatches.Ellipse: HandlerEllipse()}, markerscale=1.0, fontsize="small")

        plt.tight_layout()
        plt.subplots_adjust(left=0.14, wspace=0.2, hspace=0.2, bottom=0.19)

        print()
        #
        big_ax = fig.add_subplot(321, frameon=False)
        big_ax.set_xticks([])
        big_ax.set_yticks([])
        big_ax.set_ylabel('Agency', labelpad=50, fontsize=12)
        big_ax = fig.add_subplot(323, frameon=False)
        big_ax.set_xticks([])
        big_ax.set_yticks([])
        big_ax.set_ylabel('Beliefs', labelpad=50, fontsize=12)
        big_ax = fig.add_subplot(325, frameon=False)
        big_ax.set_xticks([])
        big_ax.set_yticks([])
        big_ax.set_ylabel('Communion', labelpad=50, fontsize=12)

        plt.savefig(f"plots/{model}/intersect_age{'_delta' if DELTA else ''}.pdf")
        plt.savefig(f"plots/{model}/intersect_age{'_delta' if DELTA else ''}.svg", format="svg")
        plt.show()

    def single_row_plot(which_dim="agency"):
        f = 1.25
        fig, axes = plt.subplots(2, 2, sharex=SHARE_AXIS, sharey=SHARE_AXIS, constrained_layout=True,
                                 figsize=(f * 4, f * 4))
        axes = axes.ravel()

        plt_data_single(df_causal, axes[0], which_dim=which_dim, titles=False)
        plt_data_single(df_fairface, axes[2], which_dim=which_dim, titles=False)
        plt_data_single(df_utkface, axes[3], which_dim=which_dim, titles=False)

        axes[0].set_title("CausalFace")
        axes[2].set_title("FairFace")
        axes[3].set_title("UTKFace")

        for i, ax in enumerate(axes.flatten()):
            ax.set_title(ax.get_title(), fontsize=11)

            if SHARE_AXIS == "all":
                ax.set_yticks([0.22, 0.24])
                ax.set_yticklabels([22.0, 24.0])
            if NO_AXIS:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_yticklabels([])
                ax.set_xticklabels([])

            if not i in [0, 2]:
                ax.set_ylabel("")
            if not i in [2, 3]:
                ax.set_xlabel("")
            ax.tick_params(axis='x', labelsize=6)  # Adjust x-axis tick label size
            ax.tick_params(axis='y', labelsize=6)  # Adjust y-axis tick label size

            # do not show anything in ax1
            axes[1].axis('off')

        handles, labels = axes[0].get_legend_handles_labels()
        print(labels)
        criteria = ["0-2", ">70", "1.2", "4.2", "female", "male", "white", "black", "asian", "x=y", "conf"]
        label_map = {"0-2": "youngest", ">70": "oldest", "1.2": "youngest", "4.2": "oldest"}
        filtered_handles = [handle for handle, label in zip(handles, labels) if label in criteria]
        filtered_labels = [label for label in labels if label in criteria]
        filtered_labels = [label_map.get(l) or l for l in filtered_labels]
        handles, labels = filtered_handles, filtered_labels

        grey_ellipse = mpatches.Ellipse((0, 0), 1, 1, **ELLIPSE_STYLE)
        handles.append(grey_ellipse)

        labels = ["women", "men", "youngest", "oldest", "Asian", "Black", "White", "pos=neg",
                  f'Cov. errors ({ELLIPSIS_SIGMA}$\sigma$)', ""]
        for h, l in zip(handles, labels):
            if l in ["youngest", "oldest", "Asian", "Black", "White"]:
                h.set_color('black')
                h.set_alpha(1.0)
        fig.legend(handles, labels, loc="center right", bbox_to_anchor=(0.93, 0.76), fancybox=True, ncol=1,
                   handler_map={mpatches.Ellipse: HandlerEllipse()}, markerscale=1.0, fontsize="small")

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.2, hspace=0.25)

        plt.savefig(f"./plots/{model}/intersect_age{'_delta' if DELTA else ''}_singlerow_{which_dim}.pdf")
        plt.show()

    full_plot()  # appendix
    single_row_plot(which_dim="agency")  # main paper


if __name__ == '__main__':
    main("clip_vit_b_32")
