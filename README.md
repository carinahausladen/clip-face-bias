# Social perception of faces in a vision-language model

This repository contains supplementary code for the paper [Social perception of faces in a vision-language model](https://arxiv.org/abs/2408.14435) authored by Carina Hausladen, Manuel Knott, Colin Camerer, and Pietro Perona.

We used Python 3.10 for this project. Please make sure to install the necessary dependencies by running:
```
pip install -r requirements.txt
```

## Project Structure

- `data/`: directory where raw data is expected
- `datasets/`: dataset implementations
- `results/`: contains csv files with precalculated cosine similarities
- `analysis/`: use precalculations from `results` folder for analysis and plots
- `plots/`, `tables/`: output folder for plots/tables
- `misc/`: miscellaneous scripts for secondary analysis
- `attributes_models.py`: contains definitions of textual content models

## Precalculate cosine similarities / CLIP inference

To precalculate cosine similarities between all pairs of images and texts, run the following script.
Per default, OpenAI's CLIP ViT-B/32 model is used. To use a different model, change the `model` variable in the script.
```
python precalculate_cossims.py
```

Our precalculate cosine similarities for CLIP ViT-B/32 can be downloaded [here](https://drive.google.com/file/d/108PuMlUc_8I9D_FNqg1S6ug__9bnCgpy/view?usp=sharing).
This folder should be placed in the `results` directory.

If you want to reproduce results with the original image datasets, please request them from the original sources.

## Reproduce Paper Results

To reproduce the results from the paper, run the scripts in the `analysis` folder.
All scripts require the precalculated cosine similarities from the `results` folder.
To resolve file paths, all scripts should be run from the root directory of this repository.

## Citation

If you find this project useful, please consider citing our preprint:
```
@inproceedings{hausladen2025social,
  author = {Hausladen, Carina I and Knott, Manuel and Camerer, Colin F and Perona, Pietro},
  title = {Social Perception of Faces in a Vision-Language Model},
  booktitle={Proceedings of the 2025 ACM Conference on Fairness, Accountability, and Transparency},
  pages={639--659},
  year={2025}
  doi = {10.1145/3715275.3732041} 
}
```

