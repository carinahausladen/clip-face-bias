"plot the three confounds smile lighting pose next to each other and compare sd"

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from attribute_models import ABCModel, StereotypeContentModel

from datasets.causalface import CausalFaceDataset

xticklabels = CausalFaceDataset.available_values
[xticklabels['smiling'].remove(value) for value in [-0.5, 0.5, 1.5]]

blank_values = {
    "pose": pd.read_csv(f'./results/clip_vit_b_32/causalface_pose_cossim_control_t2.csv').groupby("pose")[
        "cossim_<blank>"].mean().reset_index(),
    "lighting": pd.read_csv(f'./results/clip_vit_b_32/causalface_lighting_cossim_control_t2.csv').groupby("lighting")[
        "cossim_<blank>"].mean().reset_index(),
    "smiling": pd.read_csv(f'./results/clip_vit_b_32/causalface_smiling_cossim_control_t2.csv').groupby("smiling")[
        "cossim_<blank>"].mean().reset_index()
}


def preprocess_dataframe_general(df, model, attributes, col_name):
    prefix_cols = lambda atts: ["cossim_" + item for item in atts]

    for attr in attributes:
        df[f'mean_{attr}'] = df[prefix_cols(getattr(model, attr))].mean(axis=1) * 100

    cols_to_retain = [col_name] + [f'mean_{attr}' for attr in attributes]
    df = df[cols_to_retain]

    return df.groupby(col_name).mean().reset_index()


def preprocess_dataframe(df, col_name="smiling"):
    attributes_abc = ['agency_pos', 'agency_neg', 'belief_pos', 'belief_neg', 'communion_pos', 'communion_neg']
    return preprocess_dataframe_general(df, ABCModel, attributes_abc, col_name)


def preprocess_dataframe_scm(df, col_name="smiling"):
    attributes_scm = ['warm', 'comp']
    return preprocess_dataframe_general(df, StereotypeContentModel, attributes_scm, col_name)


# Loading and processing dataframes for causalface
attributes = ["smiling", "lighting", "pose"]
dfs_causal = {}

for attribute in attributes:
    df_abc_causal = pd.read_csv(f'./results/clip_vit_b_32/causalface_{attribute}_cossim_abc_t2.csv')
    df_scm_causal = pd.read_csv(f'./results/clip_vit_b_32/causalface_{attribute}_cossim_scm_t2.csv')
    df_abc_processed = preprocess_dataframe(df_abc_causal, attribute)
    df_scm_processed = preprocess_dataframe_scm(df_scm_causal, attribute)
    dfs_causal[attribute] = df_abc_processed.merge(df_scm_processed, on=attribute, how='left')


# ---------------------------- Plotting lines
def plot_data(ax, data, col_name, subtract_blank=False):
    attributes = [col for col in data.columns if col.startswith("mean_")]
    legend_mapping = {
        'agency_pos': 'A+',
        'agency_neg': 'A-',
        'belief_pos': 'B$^P$',
        'belief_neg': 'B$^C$',
        'communion_pos': 'C+',
        'communion_neg': 'C-',
        'warm': 'W',
        'comp': 'C'
    }

    primary_categories = ['agency', 'belief', 'communion', 'warmcomp']
    colors_list = ["#D81B60", "#1E88E5", "#FFC107", "#000000"]
    # colors_list = sns.color_palette("colorblind", len(primary_categories))
    category_colors = dict(zip(primary_categories, colors_list))

    # Define markers for specific attributes
    markers = {
        'pos': 'o',
        'neg': 'x',
        'warm': 's',  # square
        'comp': '^'  # triangle
    }

    for attr in attributes:
        # data[attr] = data[attr] * 100
        # Determine the marker style
        marker = 'o'  # default
        for key, val in markers.items():
            if key in attr:
                marker = val
                break
        # Determine the color based on main category
        color = 'black'  # default
        for cat in primary_categories:
            if cat in attr:
                color = category_colors[cat]
                break

        # Get the desired label from the mapping, default to the original attribute name if not found
        label = legend_mapping.get(attr.replace("mean_", ""), attr.replace("mean_", "").capitalize())

        if subtract_blank:
            data[attr] = data[attr] - blank_values[col_name]["cossim_<blank>"] * 100
        # data[attr] = data[attr] * 100

        sns.scatterplot(ax=ax, data=data, x=col_name, y=attr,
                        marker=marker,
                        label=label,
                        color=color)
        ax.plot(data[col_name], data[attr], color="grey", linewidth=0.5)

    blank_y = blank_values[col_name]["cossim_<blank>"] * 100
    if subtract_blank:
        blank_y = [0.0] * len(blank_y)
    ax.plot(blank_values[col_name][col_name], blank_y,
            color="grey", linewidth=1, linestyle="--",
            label="<blank>")

    ax.set_xlabel(f"{col_name.capitalize()}")
    # ax.set_xlabel("")
    if subtract_blank:
        ax.set_ylabel('$\Delta$ Cosine Similarity %')
    else:
        ax.set_ylabel("Cosine Similarity %")

    ax.set_xticks(ticks=xticklabels[col_name], labels=xticklabels[col_name])

    if col_name == "lighting":
        x_tick_to_gray = -1
    elif col_name == "smiling":
        x_tick_to_gray = 0
    elif col_name == "pose":
        x_tick_to_gray = 0.0
    gray_column_width = 0.5

    # Plot the gray column using axvspan
    ax.axvspan(x_tick_to_gray - (gray_column_width / 2), x_tick_to_gray + (gray_column_width / 2),
               color='gray', alpha=0.2, label='default')

    ax.legend(loc='lower right', frameon=False)

    # Format y-axis tick labels to show only two decimal places
    ax.set_yticklabels(['{:.1f}'.format(y) for y in ax.get_yticks()])
    ax.get_legend().remove()


def main(subtract_blank=False):
    fig, axes = plt.subplots(1, 3, figsize=(8, 2.8), sharey="all")
    attributes = ['smiling', 'lighting', 'pose']

    for ax, attribute in zip(axes, attributes):
        plot_data(ax, dfs_causal[attribute], attribute, subtract_blank=subtract_blank)

    # Unified legend for the whole figure
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right',
               bbox_to_anchor=(1, 0.52), prop={'size': 9},
               ncol=1, title="Social\nPerception")

    plt.tight_layout()
    plt.subplots_adjust(right=0.86)

    plt.savefig(f'plots/confounds{"_diff" if subtract_blank else ""}.pdf')
    plt.savefig(f'plots/confounds{"_diff" if subtract_blank else ""}.png', dpi=300)
    # plt.savefig(f'plots/confounds{"_diff" if subtract_blank else ""}.svg', format="svg")
    plt.show()
    plt.close()


def correlation_analysis():
    import scipy.stats as stats
    for attribute in attributes:
        df = dfs_causal[attribute]
        pairs = [('mean_agency_pos', 'mean_agency_neg'),
                 ('mean_belief_pos', 'mean_belief_neg'),
                 ('mean_communion_pos', 'mean_communion_neg')]
        corrs, ps = [], []
        for col1, col2 in pairs:
            correlation, p_value = stats.pearsonr(df[col1], df[col2])
            corrs.append(correlation)
            ps.append(p_value)
            print(f"{attribute} {col1} vs {col2}: {correlation:.2f} (p-value: {p_value:.2f})")
        print(f"{attribute} average correlation: {sum(corrs) / len(corrs):.2f}")
        print("---")


if __name__ == '__main__':
    main(subtract_blank=True)
    # main(subtract_blank=False)
    # do not run both variations on one script due to bug,
    # they share the underlying dataframe

    correlation_analysis()
