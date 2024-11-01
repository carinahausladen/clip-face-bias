import pandas as pd


MODEL = "clip_vit_b_32"

df_causal = pd.read_csv(f'./results/{MODEL}/causalface_age_cossim_control_t2.csv')

df_utk = pd.read_csv(f"results/{MODEL}/utkface_cossim_control_t2.csv")
df_utk = df_utk[df_utk['race'].isin(['White', 'Asian', 'Black'])]
df_utk = df_utk[df_utk.age > 19]
df_utk['race'] = df_utk['race'].str.lower()

df_fairface = pd.read_csv(f'./results/{MODEL}/fairface_cossim_control_t2.csv')
df_fairface.loc[df_fairface['race'] == 'East Asian', 'race'] = 'Asian'
df_fairface.loc[df_fairface['race'] == 'Southeast Asian', 'race'] = 'Asian'
df_fairface = df_fairface[df_fairface['race'].isin(['White', 'Asian', 'Black'])]
df_fairface = df_fairface[~df_fairface.age.isin(["0-2", "3-9", "10-19"])]
df_fairface['race'] = df_fairface['race'].str.lower()

# ------------ TABLE 1: calculate markedness according to Wolfe 2022
race_columns = ['cossim_white', 'cossim_black', 'cossim_asian']
gender_columns = ['cossim_male', 'cossim_female']


def calculate_percentage(df, category_columns, category_type):
    results = {}
    for col in category_columns:
        category_name = col.split('_')[1]
        filtered_df = df[df[category_type] == category_name]
        total_count = len(filtered_df)

        # check if 'cossim_<blank>' is higher than all other category cosine similarities
        mask = filtered_df['cossim_<blank>'] > filtered_df[col]
        higher_count = len(filtered_df[mask])

        percentage = (higher_count / total_count) * 100 if total_count != 0 else 0
        results[category_name] = percentage

    return results


# Calculate the percentages for the causal dataset
race_results_causal = calculate_percentage(df_causal, race_columns, 'race')
gender_results_causal = calculate_percentage(df_causal, gender_columns, 'gender')

# Calculate the percentages for the fairface dataset
race_results_fairface = calculate_percentage(df_fairface, race_columns, 'race')
gender_results_fairface = calculate_percentage(df_fairface, gender_columns, 'gender')

# Calculate the percentages for the UTK dataset
race_results_utk = calculate_percentage(df_utk, race_columns, 'race')
gender_results_utk = calculate_percentage(df_utk, gender_columns, 'gender')

# Combine the results into a single dataframe
df_markedness = pd.DataFrame({
    'Category': list(race_results_causal.keys()) + list(gender_results_causal.keys()),
    'CausalFace': list(race_results_causal.values()) + list(gender_results_causal.values()),
    'FairFace': list(race_results_fairface.values()) + list(gender_results_fairface.values()),
    'UTKFace': list(race_results_utk.values()) + list(gender_results_utk.values())
})

df_markedness.to_latex(buf=f"results/{MODEL}/markedness.tex", index=False, float_format="%.2f")

########################################################################################################################
# COSSIM
########################################################################################################################

"table fo the average cossims per race and gender"
import os

##os.chdir("..") # set working dir to project root
os.getcwd()
import pandas as pd


def preprocess_df(df_in, df_causal):
    df_in['race'] = df_in['race'].apply(lambda x: 'Asian' if 'Asian' in x else x)
    df_in['race'] = df_in['race'].str.lower()
    df_in['gender'] = df_in['gender'].str.lower()

    unique_races_causal = df_causal['race'].unique()
    unique_gender_causal = df_causal['gender'].unique()
    selected_rows = df_in[(df_in['race'].isin(unique_races_causal)) &
                          (df_in['gender'].isin(unique_gender_causal))]
    df_new = selected_rows.copy()
    return df_new


df_causal_scm = pd.read_csv(f'./results/{MODEL}/causalface_age_cossim_scm_t2.csv')
df_causal_abc = pd.read_csv(f'./results/{MODEL}/causalface_age_cossim_abc_t2.csv')

df_fairface_scm = pd.read_csv(f'./results/{MODEL}/fairface_cossim_scm_t2.csv')
df_fairface_scm = preprocess_df(df_fairface_scm, df_causal_scm)
df_fairface_abc = pd.read_csv(f'./results/{MODEL}/fairface_cossim_abc_t2.csv')
df_fairface_abc = preprocess_df(df_fairface_abc, df_causal_scm)

df_utk_scm = pd.read_csv(f"results/{MODEL}/utkface_cossim_scm_t2.csv")
df_utk_scm = preprocess_df(df_utk_scm, df_causal_scm)
df_utk_abc = pd.read_csv(f'./results/{MODEL}/utkface_cossim_abc_t2.csv')
df_utk_abc = preprocess_df(df_utk_abc, df_causal_scm)

### additional age filtering
df_fairface_scm = df_fairface_scm[~df_fairface_scm.age.isin(["0-2", "3-9", "10-19"])]
df_fairface_abc = df_fairface_abc[~df_fairface_abc.age.isin(["0-2", "3-9", "10-19"])]
df_utk_scm = df_utk_scm[df_utk_scm.age > 19]
df_utk_abc = df_utk_abc[df_utk_abc.age > 19]

# ------------------------ get attributes
from attribute_models import ABCModel, StereotypeContentModel


def get_attribute_cols(group):
    return ["cossim_" + attribute for attribute in group]


warmth = get_attribute_cols(StereotypeContentModel.warm)
competence = get_attribute_cols(StereotypeContentModel.comp)
agency_pos = get_attribute_cols(ABCModel.agency_pos)
agency_neg = get_attribute_cols(ABCModel.agency_neg)
belief_pos = get_attribute_cols(ABCModel.belief_pos)
belief_neg = get_attribute_cols(ABCModel.belief_neg)
communion_pos = get_attribute_cols(ABCModel.communion_pos)
communion_neg = get_attribute_cols(ABCModel.communion_neg)


# ------------------------ compute means
def compute_attribute_means(df, cols, grouping):
    return df.groupby([grouping])[cols].mean().mean(axis=1)  # , 'gender'


def comp_all_means(df_scm, df_abc, grouping):
    all_means = pd.DataFrame({
        'warmth': compute_attribute_means(df_scm, warmth, grouping),
        'competence': compute_attribute_means(df_scm, competence, grouping),
        'agency_pos': compute_attribute_means(df_abc, agency_pos, grouping),
        'belief_pos': compute_attribute_means(df_abc, belief_pos, grouping),
        'communion_pos': compute_attribute_means(df_abc, communion_pos, grouping),
    })
    return all_means


m_ff_race = comp_all_means(df_fairface_scm, df_fairface_abc, "race").mean(axis=1)
m_ff_gender = comp_all_means(df_fairface_scm, df_fairface_abc, "gender").mean(axis=1)
result_ff = pd.concat([m_ff_race, m_ff_gender], axis=0)

m_cf_race = comp_all_means(df_causal_scm, df_causal_abc, "race").mean(axis=1)
m_cf_gender = comp_all_means(df_causal_scm, df_causal_abc, "gender").mean(axis=1)
result_cf = pd.concat([m_cf_race, m_cf_gender], axis=0)

m_uf_race = comp_all_means(df_utk_scm, df_utk_abc, "race").mean(axis=1)
m_uf_gender = comp_all_means(df_utk_scm, df_utk_abc, "gender").mean(axis=1)
result_uf = pd.concat([m_uf_race, m_uf_gender], axis=0)

# ------------------------ merge means
result_cf = pd.DataFrame(result_cf, columns=['CausalFace'], index=['asian', 'black', 'white', 'female', 'male'])
result_ff = pd.DataFrame(result_ff, columns=['FairFace'], index=['asian', 'black', 'white', 'female', 'male'])
result_uf = pd.DataFrame(result_uf, columns=['UTKFace'], index=['asian', 'black', 'white', 'female', 'male'])

df_cossim = pd.concat([result_cf, result_ff, result_uf], axis=1) * 100
df_cossim.reset_index(inplace=True)
df_cossim = df_cossim.rename(columns={'index': 'Category'}, inplace=False)
latex_table = df_cossim.to_latex(float_format="%.2f", multirow=True)
with open(f"results/{MODEL}/means.tex", "w") as file:
    file.write(latex_table)

########################################################################################################################
# overall table
########################################################################################################################
merged_df = pd.merge(df_markedness, df_cossim, on="Category")

latex_code = merged_df.to_latex(index=False, column_format="lcccccc", float_format="%.2f")
print(latex_code)
