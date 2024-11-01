"""
Script to run CLIP inferences and calculate cosine similarities for a datasets with given text embeddings.
Save to csv under results/ directory.
"""

import os
from collections import OrderedDict
from typing import List

import torch
import clip
from tqdm import tqdm

from datasets import get_dataset
from attribute_models import StereotypeContentModel, ABCModel, Control

device = "cuda" if torch.cuda.is_available() else "cpu"

TEMPLATES = {
    1: [
        "A photo of a {attribute} person."
    ],
    2: [
        "A photo of a {attribute} person.",
        "A {attribute} person.",
        "This is a {attribute} person.",
        "cropped face photo of a {attribute} person"
    ]
}


@torch.no_grad()
def calculate_img_embedding(dataset, img_path: str, model, preprocess):
    img = dataset.get_image_by_path(img_path)
    img = preprocess(img).unsqueeze(0).to(device)
    emb = model.encode_image(img).squeeze()
    return emb.detach()


@torch.no_grad()
def calculate_clip_text_embeddings(words: List[str], templates: List[str], model):
    all_prompts = []
    all_prompts.extend([phrase.format(attribute="" if word == "<blank>" else word)
                        for word in words for phrase in templates])
    all_prompts = [p.replace("  ", " ") for p in all_prompts]
    text_tokens = clip.tokenize(all_prompts).to(device)
    emb = model.encode_text(text_tokens).detach()
    return emb, words


def create_processed_csv(dataset: str,
                         words_to_match: List[str],
                         output_suffix: str = "",
                         model_name: str = "ViT-B/32",
                         templates_set=2,
                         ):

    templates = TEMPLATES[templates_set]

    model, preprocess = clip.load(model_name, device=device)
    text_embs, words = calculate_clip_text_embeddings(words=words_to_match, templates=templates, model=model)

    ds = get_dataset(dataset)
    meta_df = ds.meta_df()

    data = OrderedDict()
    for word in words:
        data[f"cossim_{word}"] = []

    for img_path in tqdm(meta_df["img_path"].values, desc=f"Calculating image embeddings "
                                                          f"({dataset}, {output_suffix}, templates={templates_set})"):

        img_emb = calculate_img_embedding(ds, img_path, model, preprocess)
        cos_sims = torch.nn.functional.cosine_similarity(img_emb, text_embs)
        cos_sims = cos_sims.reshape(-1, len(templates)).mean(axis=1)  # average over prompts per attribute
        for word, cossim in zip(words, cos_sims):
            data[f"cossim_{word}"].append(cossim.item())

    df = meta_df.copy()
    for key, vals in data.items():
        df[key] = vals  # add cosim columns to df

    os.makedirs("results", exist_ok=True)
    clip_folder_name = "clip_" + model_name.lower().replace("/", "_").replace("-", "_")
    os.makedirs(f"results/{clip_folder_name}", exist_ok=True)
    filepath = f"results/{clip_folder_name}/{dataset}_cossim_{output_suffix}_t{templates_set}.csv"
    print("Saving results csv to ", filepath)
    df.to_csv(filepath, index=False)


if __name__ == '__main__':

    model_name = "ViT-B/32"

    # Fairface
    create_processed_csv(model_name=model_name, dataset="fairface",
                         words_to_match=StereotypeContentModel.all_attributes,
                         output_suffix=StereotypeContentModel.name, templates_set=2)
    create_processed_csv(model_name=model_name, dataset="fairface", words_to_match=ABCModel.all_attributes,
                         output_suffix=ABCModel.name, templates_set=2)
    exit()

    create_processed_csv(model_name=model_name, dataset="causalface_age", words_to_match=Control.all_attributes, output_suffix=Control.name, templates_set=2)
    create_processed_csv(model_name=model_name, dataset="causalface_age", words_to_match=StereotypeContentModel.all_attributes, output_suffix=StereotypeContentModel.name, templates_set=2)
    create_processed_csv(model_name=model_name, dataset="causalface_age", words_to_match=ABCModel.all_attributes, output_suffix=ABCModel.name, templates_set=2)

    create_processed_csv(model_name=model_name, dataset="causalface_pose", words_to_match=Control.all_attributes, output_suffix=Control.name, templates_set=2)
    create_processed_csv(model_name=model_name, dataset="causalface_pose", words_to_match=StereotypeContentModel.all_attributes, output_suffix=StereotypeContentModel.name, templates_set=2)
    create_processed_csv(model_name=model_name, dataset="causalface_pose", words_to_match=ABCModel.all_attributes, output_suffix=ABCModel.name, templates_set=2)

    create_processed_csv(model_name=model_name, dataset="causalface_lighting", words_to_match=Control.all_attributes, output_suffix=Control.name, templates_set=2)
    create_processed_csv(model_name=model_name, dataset="causalface_lighting", words_to_match=StereotypeContentModel.all_attributes, output_suffix=StereotypeContentModel.name, templates_set=2)
    create_processed_csv(model_name=model_name, dataset="causalface_lighting", words_to_match=ABCModel.all_attributes, output_suffix=ABCModel.name, templates_set=2)

    create_processed_csv(model_name=model_name, dataset="causalface_smiling", words_to_match=Control.all_attributes, output_suffix=Control.name, templates_set=2)
    create_processed_csv(model_name=model_name, dataset="causalface_smiling", words_to_match=StereotypeContentModel.all_attributes, output_suffix=StereotypeContentModel.name, templates_set=2)
    create_processed_csv(model_name=model_name, dataset="causalface_smiling", words_to_match=ABCModel.all_attributes, output_suffix=ABCModel.name, templates_set=2)

    # brightness adjustments
    # First, misc/create_causalface_brightness.py needs to be run to extend the dataset.
    # create_processed_csv(model_name="ViT-B/32", dataset="causalface_brightness", words_to_match=Control.all_attributes, output_suffix=Control.name, templates_set=2)
    # create_processed_csv(model_name="ViT-B/32", dataset="causalface_brightness",  words_to_match=StereotypeContentModel.all_attributes,  output_suffix=StereotypeContentModel.name, templates_set=2)
    # create_processed_csv(model_name="ViT-B/32", dataset="causalface_brightness", words_to_match=ABCModel.all_attributes, output_suffix=ABCModel.name, templates_set=2)

    # Fairface
    create_processed_csv(model_name=model_name, dataset="fairface", words_to_match=Control.all_attributes, output_suffix=Control.name, templates_set=2)
    create_processed_csv(model_name=model_name, dataset="fairface", words_to_match=StereotypeContentModel.all_attributes, output_suffix=StereotypeContentModel.name, templates_set=2)
    create_processed_csv(model_name=model_name, dataset="fairface", words_to_match=ABCModel.all_attributes, output_suffix=ABCModel.name, templates_set=2)

    # UTKFace
    create_processed_csv(model_name=model_name, dataset="utkface", words_to_match=Control.all_attributes, output_suffix=Control.name, templates_set=2)
    create_processed_csv(model_name=model_name, dataset="utkface", words_to_match=StereotypeContentModel.all_attributes, output_suffix=StereotypeContentModel.name, templates_set=2)
    create_processed_csv(model_name=model_name, dataset="utkface", words_to_match=ABCModel.all_attributes, output_suffix=ABCModel.name, templates_set=2)

