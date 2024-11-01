"Visualizing the text2text embeddings in a 3D space."

import clip
import torch
from attribute_models import StereotypeContentModel, ABCModel
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

texts = StereotypeContentModel.warm + StereotypeContentModel.comp + ABCModel.all_attributes
text_inputs = torch.cat([clip.tokenize(t).to(device) for t in texts])
text_features = model.encode_text(text_inputs)

pca = PCA(n_components=3)
reduced_features = pca.fit_transform(text_features.detach().cpu().numpy())
df_reduced_features = pd.DataFrame(reduced_features, columns=['Feature1', 'Feature2', 'Feature3']).assign(Texts=texts)


def determine_model(text):
    if text in StereotypeContentModel.warm:
        return "warm"
    elif text in StereotypeContentModel.comp:
        return "comp"
    elif text in ABCModel.agency_pos:
        return "agency_pos"
    elif text in ABCModel.agency_neg:
        return "agency_neg"
    elif text in ABCModel.belief_pos:
        return "belief_pos"
    elif text in ABCModel.belief_neg:
        return "belief_neg"
    elif text in ABCModel.communion_neg:
        return "communion_neg"
    elif text in ABCModel.communion_pos:
        return "communion_pos"
    else:
        return "Unknown"


model_marker_dict = {
    'warm': 'W',
    'comp': 'C',
    'agency_pos': 'a+',
    'agency_neg': 'a-',
    'belief_pos': 'bP',
    'belief_neg': 'bC',
    'communion_pos': 'c+',
    'communion_neg': 'c-'
}

df_reduced_features['Model'] = df_reduced_features['Texts'].apply(determine_model)
models = df_reduced_features['Model'].unique()

color_dict = {
    'warm': '#1E88E5',
    'comp': '#1E88E5',
    'agency_pos': '#D81B60',
    'agency_neg': '#D81B60',
    'belief_pos': '#FFC107',
    'belief_neg': '#FFC107',
    'communion_pos': '#FE6100',
    'communion_neg': '#FE6100'
}

fig = plt.figure(figsize=(2.51, 2.51))
ax = fig.add_subplot(111, projection='3d')

# Iterate only over the models without zipping with colors
for model in models:
    subset = df_reduced_features[df_reduced_features['Model'] == model]
    center = subset[['Feature1', 'Feature2', 'Feature3']].mean().values

    # Fetch the color for the current model from the color_dict
    model_color = color_dict[model]
    marker = model_marker_dict[model]

    # Annotate each point in the subset with the desired marker
    for index, row in subset.iterrows():
        ax.text(row['Feature1'], row['Feature2'], row['Feature3'], marker, color=model_color,
                fontsize=6, ha='center', alpha=.5, va='center')

        ax.plot([row['Feature1'], center[0]],
                [row['Feature2'], center[1]],
                [row['Feature3'], center[2]], color=model_color,
                linestyle=':', linewidth=0.5)

    # Annotate the center
    ax.text(center[0], center[1], center[2], marker, color=model_color,
            fontsize=9, ha='center', va='center', fontweight='bold')

ax.set_xlim([df_reduced_features['Feature1'].min(), 2.2])
ax.set_ylim([df_reduced_features['Feature2'].min(), 2])
ax.set_zlim([df_reduced_features['Feature3'].min(), 1])

ax.xaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(1))
ax.zaxis.set_major_locator(MultipleLocator(1))

plt.savefig("plots/3D_plot.pdf", bbox_inches='tight')
plt.show()
