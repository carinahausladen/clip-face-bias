"this script calculates skewness and max skew; according to Geyik (2019)"

import os

# os.chdir("..") # set working dir to project root
os.getcwd()

import pandas as pd
import numpy as np


def main(model: str):
    df_causal_scm = pd.read_csv(f'./results/{model}/causalface_age_cossim_scm_t2.csv')
    df_causal_abc = pd.read_csv(f'./results/{model}/causalface_age_cossim_abc_t2.csv')

    df_fairface_scm = pd.read_csv(f'./results/{model}/fairface_cossim_scm_t2.csv')
    df_fairface_abc = pd.read_csv(f'./results/{model}/fairface_cossim_abc_t2.csv')

    df_utk_scm = pd.read_csv(f"results/{model}/utkface_cossim_scm_t2.csv")
    df_utk_abc = pd.read_csv(f"results/{model}/utkface_cossim_abc_t2.csv")

    def preprocess_data(df):
        df['race'] = df['race'].apply(lambda x: 'asian' if 'Asian' in x else x.lower())
        df['gender'] = df['gender'].str.lower()
        df = df[df['race'].isin(['white', 'asian', 'black'])]
        num_samples_per_group = len(df) // (len(df['race'].unique()) * len(df['gender'].unique()))
        samples = []
        for race in df['race'].unique():
            for gender in df['gender'].unique():
                group = df[(df['race'] == race) & (df['gender'] == gender)]
                sample = group.sample(n=min(num_samples_per_group, len(group)), random_state=42)
                samples.append(sample)
        df_fair = pd.concat(samples).reset_index(drop=True)
        return df_fair

    df_fairface_scm = preprocess_data(df_fairface_scm)
    df_fairface_scm = df_fairface_scm[~df_fairface_scm.age.isin(["0-2", "3-9", "10-19"])]
    df_fairface_abc = preprocess_data(df_fairface_abc)
    df_fairface_abc = df_fairface_abc[~df_fairface_abc.age.isin(["0-2", "3-9", "10-19"])]

    df_utk_scm = preprocess_data(df_utk_scm)
    df_utk_scm = df_utk_scm[df_utk_scm.age > 19]
    df_utk_abc = preprocess_data(df_utk_abc)
    df_utk_abc = df_utk_abc[df_utk_abc.age > 19]

    # -------------- PROPORTIONS
    race_proportions = {
        'asian': 1 / 3,
        'white': 1 / 3,
        'black': 1 / 3
    }
    gender_proportions = {
        'male': 1 / 2,
        'female': 1 / 2
    }

    # -------------- attribute columns
    from attribute_models import ABCModel, StereotypeContentModel

    def get_columns_for_attribute_group(group):
        return ["cossim_" + attribute for attribute in group]

    # Getting columns for each attribute group
    warmth = get_columns_for_attribute_group(StereotypeContentModel.warm)
    competence = get_columns_for_attribute_group(StereotypeContentModel.comp)
    agency_pos = get_columns_for_attribute_group(ABCModel.agency_pos)
    agency_neg = get_columns_for_attribute_group(ABCModel.agency_neg)
    belief_pos = get_columns_for_attribute_group(ABCModel.belief_pos)
    belief_neg = get_columns_for_attribute_group(ABCModel.belief_neg)
    communion_pos = get_columns_for_attribute_group(ABCModel.communion_pos)
    communion_neg = get_columns_for_attribute_group(ABCModel.communion_neg)

    # ----------------- functions: skew at k
    def skew_at_k(df_in, dimension, attribute_col, k, desired_props):
        """
        Calculate Skew@k for a single attribute e.g. "friendly"
        """

        df_in['avg_dimension'] = df_in[dimension].mean(axis=1)
        df_in = df_in[[attribute_col, 'avg_dimension']]
        top_k = df_in.nlargest(k, 'avg_dimension')  # Rank images based on similarity
        proportions = top_k[attribute_col].value_counts(
            normalize=True)  # Compute actual proportion of images with each attribute category in the top k
        skew_values = {}
        for category, desired_proportion in desired_props.items():
            actual_proportion = proportions.get(category, 0)  # Use 0 if category is not in top_k
            skew_values[category] = np.log(actual_proportion / desired_proportion)

        return skew_values

    skew_at_k(df_causal_scm, warmth, 'gender', k=100, desired_props=gender_proportions)

    def skews_df_dimension(datasets, attribute_col, scm_cols, abc_cols, desired_props, k=1000):
        """
        Calculates skews for both datasets and all dimensions.
        """
        all_skews = {}

        for dataset_name, df in datasets.items():
            cossim_cols = scm_cols if 'SCM' in dataset_name else abc_cols
            dataset_results = {}

            for dimension, cols in cossim_cols.items():
                dimension_skew = skew_at_k(df, cols, attribute_col, k, desired_props)
                dataset_results[dimension] = dimension_skew

            all_skews[dataset_name] = dataset_results

        return all_skews

    datasets = {
        'CausalFace_SCM': df_causal_scm,
        'CausalFace_ABC': df_causal_abc,
        'FairFace_SCM': df_fairface_scm,
        'FairFace_ABC': df_fairface_abc,
        'UTKFace_SCM': df_utk_scm,
        'UTKFace_ABC': df_utk_abc,
    }
    scm_cols = {
        'W': warmth,
        'C': competence
    }
    abc_cols = {
        'A+': agency_pos,
        'A-': agency_neg,
        'B+': belief_pos,
        'B-': belief_neg,
        'C+': communion_pos,
        'C-': communion_neg
    }

    def convert_skews_to_df(skew_data):
        rows = []

        for dataset_name, dimensions in skew_data.items():
            for dimension, attributes in dimensions.items():
                for attribute, value in attributes.items():
                    rows.append({
                        'which_df': dataset_name,
                        'dimension': dimension,
                        'attribute': attribute,
                        'value': value
                    })

        return pd.DataFrame(rows)

    ## RACE
    df_skew_race = skews_df_dimension(datasets, 'race', scm_cols, abc_cols, race_proportions)
    df_skew_race = convert_skews_to_df(df_skew_race)
    df_skew_race['which_df'] = df_skew_race['which_df'].str.split('_').str[0]

    ## GENDER
    df_skew_gender = skews_df_dimension(datasets, 'gender', scm_cols, abc_cols, gender_proportions)
    df_skew_gender = convert_skews_to_df(df_skew_gender)
    df_skew_gender['which_df'] = df_skew_gender['which_df'].str.split('_').str[0]

    df_skew = pd.concat([df_skew_race, df_skew_gender], ignore_index=True)

    df_skew['dimension'] = df_skew['dimension'].replace(['W', 'C'], ['warm', 'comp'])

    combined_csv_path = f'./results/{model}/skew.csv'
    df_skew.to_csv(combined_csv_path, index=False)


if __name__ == '__main__':
    main("clip_vit_b_32")
