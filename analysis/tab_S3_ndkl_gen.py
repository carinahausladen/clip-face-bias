"calcualtes NDKL according to Geyik 2019"
import warnings


def main(model: str):
    import pandas as pd
    df_causal_scm = pd.read_csv(f'./results/{model}/causalface_age_cossim_scm_t2.csv')
    df_causal_abc = pd.read_csv(f'./results/{model}/causalface_age_cossim_abc_t2.csv')

    df_fairface_scm = pd.read_csv(f'./results/{model}/fairface_cossim_scm_t2.csv')
    df_fairface_abc = pd.read_csv(f'./results/{model}/fairface_cossim_abc_t2.csv')

    df_utk_scm = pd.read_csv(f'./results/{model}/utkface_cossim_scm_t2.csv')
    df_utk_abc = pd.read_csv(f'./results/{model}/utkface_cossim_abc_t2.csv')

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

    # -------------- attribute columns
    from attribute_models import ABCModel, StereotypeContentModel

    def get_columns_for_attribute_group(group):
        return ["cossim_" + attribute for attribute in group]

    warmth = get_columns_for_attribute_group(StereotypeContentModel.warm)
    competence = get_columns_for_attribute_group(StereotypeContentModel.comp)
    agency_pos = get_columns_for_attribute_group(ABCModel.agency_pos)
    agency_neg = get_columns_for_attribute_group(ABCModel.agency_neg)
    belief_pos = get_columns_for_attribute_group(ABCModel.belief_pos)
    belief_neg = get_columns_for_attribute_group(ABCModel.belief_neg)
    communion_pos = get_columns_for_attribute_group(ABCModel.communion_pos)
    communion_neg = get_columns_for_attribute_group(ABCModel.communion_neg)

    dimensions_scm = {
        'W': warmth,
        'C': competence}

    dimensions_abc = {
        'A+': agency_pos,
        'A-': agency_neg,
        'B+': belief_pos,
        'B-': belief_neg,
        'C+': communion_pos,
        'C-': communion_neg}

    # ---------------------------------------- NDKL
    import pandas as pd
    import numpy as np
    from joblib import Parallel, delayed

    def KL_divergence(D1, D2):
        return np.sum([D1[key] * np.log(D1[key] / D2[key]) for key in D1 if key in D2 and D1[key] != 0])

    def NDKL(df, dimension, attribute, Dr):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            df['avg_dimension'] = df[dimension].mean(axis=1)
            df = df[[attribute, 'avg_dimension']]
            df_sorted = df.sort_values(by="avg_dimension", ascending=False)

            Z = np.sum(1 / np.log2(np.arange(1, len(df_sorted) + 1) + 1))

            cumsum_counts = pd.get_dummies(df_sorted[attribute]).cumsum()
            cumsum_fracs = cumsum_counts.div(cumsum_counts.sum(axis=1), axis=0)

            for attr_name, attr_val in Dr.items():
                cumsum_fracs[f"desired_{attr_name}"] = attr_val
                cumsum_fracs[f"KLD_{attr_name}"] = (cumsum_fracs[attr_name] * np.log(
                    cumsum_fracs[attr_name] / cumsum_fracs[f"desired_{attr_name}"])).fillna(0.0)
            KLD = cumsum_fracs[[f"KLD_{attr_name}" for attr_name in Dr]].sum(axis=1)
            NDKL = (1 / np.log2(np.arange(1, len(KLD) + 1) + 1)) * KLD
        return NDKL.sum() / Z

    def NDKL_dimension(df, dimensions, attribute, Dr):
        dimension_dict = {}
        for dimension, cossims in dimensions.items():
            NDKL_values = NDKL(df, cossims, attribute, Dr)
            dimension_dict[dimension] = NDKL_values
        return dimension_dict

    # ---------------------------------------- calculate

    Dr_gender = {'male': 0.5, 'female': 0.5}
    Dr_race = {'asian': 1 / 3, 'white': 1 / 3, 'black': 1 / 3}

    causal_gender_scm = NDKL_dimension(df_causal_scm, dimensions_scm, 'gender', Dr_gender)
    causal_gender_abc = NDKL_dimension(df_causal_abc, dimensions_abc, 'gender', Dr_gender)
    causal_race_scm = NDKL_dimension(df_causal_scm, dimensions_scm, 'race', Dr_race)
    causal_race_abc = NDKL_dimension(df_causal_abc, dimensions_abc, 'race', Dr_race)

    fairface_gender_scm = NDKL_dimension(df_fairface_scm, dimensions_scm, 'gender', Dr_gender)
    fairface_gender_abc = NDKL_dimension(df_fairface_abc, dimensions_abc, 'gender', Dr_gender)
    fairface_race_scm = NDKL_dimension(df_fairface_scm, dimensions_scm, 'race', Dr_race)
    fairface_race_abc = NDKL_dimension(df_fairface_abc, dimensions_abc, 'race', Dr_race)

    utk_gender_scm = NDKL_dimension(df_utk_scm, dimensions_scm, 'gender', Dr_gender)
    utk_gender_abc = NDKL_dimension(df_utk_abc, dimensions_abc, 'gender', Dr_gender)
    utk_race_scm = NDKL_dimension(df_utk_scm, dimensions_scm, 'race', Dr_race)
    utk_race_abc = NDKL_dimension(df_utk_abc, dimensions_abc, 'race', Dr_race)

    # ---------------------------------------- write CSV

    def generate_df(dict, which_df, which_attribute):
        df = pd.DataFrame(list(dict.items()), columns=['dimension', 'NDKL'])
        df['which_df'] = which_df
        df['attribute'] = which_attribute

        return df

    df_list = [
        generate_df(causal_gender_scm, 'causalface', 'gender'),
        generate_df(causal_gender_abc, 'causalface', 'gender'),
        generate_df(causal_race_scm, 'causalface', 'race'),
        generate_df(causal_race_abc, 'causalface', 'race'),
        generate_df(fairface_gender_scm, 'fairface', 'gender'),
        generate_df(fairface_gender_abc, 'fairface', 'gender'),
        generate_df(fairface_race_scm, 'fairface', 'race'),
        generate_df(fairface_race_abc, 'fairface', 'race'),
        generate_df(utk_gender_scm, 'utkface', 'gender'),
        generate_df(utk_gender_abc, 'utkface', 'gender'),
        generate_df(utk_race_scm, 'utkface', 'race'),
        generate_df(utk_race_abc, 'utkface', 'race')
    ]

    combined_df = pd.concat(df_list)

    combined_csv_path = f'./results/{model}/NDKL.csv'
    combined_df.to_csv(combined_csv_path, index=False)


if __name__ == '__main__':
    main("clip_vit_b_32")
