"tables NDKL, skew and Maxskew according to Geyik (2019)"

# Make sure to first run tab_S3_ndkl_gen.py, tab_S3_skew_gen.py!

import pandas as pd

MODEL = "clip_vit_b_32"

df_combined = pd.read_csv(f'./results/{MODEL}/NDKL.csv')
df_combined = df_combined.groupby(['attribute', 'which_df']).mean().reset_index()
df_combined = df_combined.pivot(index='attribute', columns='which_df', values='NDKL').reset_index()
df_combined.to_latex(buf="tables/ndkl.tex", index=False, float_format="%.2f")

df_skew = pd.read_csv(f'./results/{MODEL}/skew.csv')
df_skew = df_skew.groupby(['attribute', 'which_df']).mean().reset_index()
df_skew = df_skew.pivot(index='attribute', columns='which_df', values='value').reset_index()
df_skew.to_latex(buf="tables/skew.tex", index=False, float_format="%.2f")

df_maxskew = pd.read_csv(f'./results/{MODEL}/skew.csv')
df_maxskew['attribute'] = df_maxskew['attribute'].replace(['asian', 'white', 'black'], 'race')
df_maxskew['attribute'] = df_maxskew['attribute'].replace(['male', 'female'], 'gender')
df_maxskew = df_maxskew.groupby(['which_df', 'dimension', 'attribute']).value.max().reset_index()
df_maxskew = df_maxskew.groupby(['attribute', 'which_df']).mean().reset_index()
df_maxskew = df_maxskew.pivot(index='attribute', columns='which_df', values='value').reset_index()
df_maxskew.to_latex(buf="tables/maxskew.tex", index=False, float_format="%.2f")
