"calculates correlations for smiling/lighting/pose with the social dimensions."
'Write results to Table S.4'

import pandas as pd
import scipy.stats as stats
from attribute_models import ABCModel, StereotypeContentModel


def process_list(df, attribute_list, id_var, value_name):
    columns = [col for col in df.columns for item in attribute_list if col.endswith(item)]
    df_melted = df.melt(id_vars=[id_var], value_vars=columns, value_name=value_name).drop('variable', axis=1)
    correlation, p_value = stats.pearsonr(df_melted[id_var], df_melted[value_name])
    return correlation, p_value


def load_and_process_data(file_causal, file_control):
    df_causal = pd.read_csv(file_causal)
    df_blank = pd.read_csv(file_control).set_index("img_path")["cossim_<blank>"]
    df_causal.set_index("img_path", inplace=True)
    cossim_cols = [col for col in df_causal.columns if col.startswith("cossim_")]
    df_causal[cossim_cols] = df_causal[cossim_cols].subtract(df_blank, axis=0)
    df_causal.reset_index(inplace=True)
    return df_causal


df_abc_causal_smiling = load_and_process_data('./results/causalface_smiling_cossim_abc_t2.csv',
                                              './results/causalface_smiling_cossim_control_t2.csv')
df_abc_causal_lighting = load_and_process_data('./results/causalface_lighting_cossim_abc_t2.csv',
                                               './results/causalface_lighting_cossim_control_t2.csv')
df_abc_causal_pose = load_and_process_data('./results/causalface_pose_cossim_abc_t2.csv',
                                           './results/causalface_pose_cossim_control_t2.csv')

# lighting
attributes_list = ABCModel.agency_pos + ABCModel.belief_pos + ABCModel.communion_pos + StereotypeContentModel.warm + StereotypeContentModel.comp + ABCModel.agency_neg + ABCModel.belief_neg + ABCModel.communion_neg
df = df_abc_causal_lighting
corr_lighting, p_val_lighting = process_list(df, attributes_list, 'lighting', 'score')

# pose
df = df_abc_causal_pose
df = df_abc_causal_pose[df_abc_causal_pose['pose'] < 0]
corr_pose_left, p_val_pose_left = process_list(df, attributes_list, 'pose', 'score')

df = df_abc_causal_pose[df_abc_causal_pose['pose'] > 0]
corr_pose_right, p_val_pose_right = process_list(df, attributes_list, 'pose', 'score')

# smiling
df = df_abc_causal_smiling
# smiling pos
attributes_list = ABCModel.agency_pos + ABCModel.belief_pos + ABCModel.communion_pos + StereotypeContentModel.warm + StereotypeContentModel.comp
corr_smiling_pos, p_val_smiling_pos = process_list(df, attributes_list, 'smiling', 'score')
# smiling neg
attributes_list = ABCModel.agency_neg + ABCModel.communion_neg
corr_smiling_neg, p_val_smiling_neg = process_list(df, attributes_list, 'smiling', 'score')
# smiling belief
attributes_list = ABCModel.belief_neg
corr_smiling_belief_neg, p_val_smiling_belief_neg = process_list(df, attributes_list, 'smiling', 'score')

# generate output
data = [
    {"Setting": "Lighting", "Correlation": corr_lighting, "P-Value": p_val_lighting},
    {"Setting": "Pose right", "Correlation": corr_pose_right, "P-Value": p_val_pose_right},
    {"Setting": "Pose left", "Correlation": corr_pose_left, "P-Value": p_val_pose_left},
    {"Setting": "Smiling -", "Correlation": corr_smiling_neg, "P-Value": p_val_smiling_neg},
    {"Setting": "Smiling +/Progressive", "Correlation": corr_smiling_pos, "P-Value": p_val_smiling_pos},
    {"Setting": "Smiling Conservative", "Correlation": corr_smiling_belief_neg, "P-Value": p_val_smiling_belief_neg}
]
df_results = pd.DataFrame(data)
df_results['P-Value'] = df_results['P-Value'].apply(lambda x: "<0.01" if x < 0.01 else ">0.01")
df_results['Correlation'] = pd.DataFrame(df_results['Correlation']).round(2)
print(df_results)
latex_output = df_results.to_latex(index=False, float_format="%.2f")
print(latex_output)
