# Social perception of faces in a vision-language model

This repository contains supplementary code for the paper [Social perception of faces in a vision-language model](https://arxiv.org/abs/2408.14435) authored by Carina Hausladen, Manuel Knott, Colin Camerer, and Pietro Perona.

## Project Structure

- `data/`: directory where raw data is expected
- `datasets/`: dataset implementations
- `results/`: contains csv files with precalculated cosine similarities
- `analysis/`: use precalculations from `results` folder for analysis and plots
- `plots/`: output folder for plots
- `misc/`: miscellaneous scripts for secondary analysis
- `attributes_models.py`: contains definitions of textual content models

## Precalculate cosine similarities / CLIP inference

To precalculate cosine similarities between all pairs of images and texts, run the following script.
Per default, OpenAI's CLIP ViT-B/32 model is used. To use a different model, change the `model` variable in the script.
```
python precalculate_cossims.py
```
We include the precalculated csv's but cannot share the original image datasets.

## Reproduce Paper Results

To reproduce the results from the paper, run the scripts in the `analysis` folder.

## Citation

If you find this project useful, please consider citing our preprint:
```
@article{hausladen2024social,
  title={Social perception of faces in a vision-language model},
  author={Hausladen, Carina I and Knott, Manuel and Camerer, Colin F and Perona, Pietro},
  journal={arXiv preprint arXiv:2408.14435},
  year={2024}
}
```

