import pandas as pd

INDEX_COLS = ['img_id', 'img_path', 'seed', 'demo', 'race', 'gender']
ATTRIBUTE_COLS = ['<blank>', 'white', 'black', 'asian', 'male', 'female', 'young', 'old',
                  'warm', 'comp', 'a+', 'a-', 'b+', 'b-', 'c+', 'c-']


def _get_groupby_scheme(by):
    return ["seed"] + by.split("_")


def _load_causalface_data(model: str, subset: str, attribute_model: str):
    filepath = f"results/{model}/causalface_{subset}_cossim_{attribute_model}_t2.csv"
    return pd.read_csv(filepath)


def _groupby_seed_demo(df, cols, groupby_scheme="gender_race", agg="std"):
    groupby_cols = _get_groupby_scheme(groupby_scheme)
    if agg == "std":
        df = df.groupby(groupby_cols)[cols].std()
    elif agg == "range":
        df = df.groupby(groupby_cols)[cols].max() - df.groupby(groupby_cols)[cols].min()
    else:
        raise NotImplementedError(f"Unknown aggregation method: {agg}")
    return df


def load_and_preprocess_control(model, subset):
    control_cols = ['cossim_<blank>', 'cossim_white', 'cossim_black', 'cossim_asian',
                    'cossim_male', 'cossim_female', 'cossim_young', 'cossim_old']
    df = _load_causalface_data(model, subset, "control")

    df.rename(columns={col: col[7:] for col in control_cols}, inplace=True)
    return df


def load_and_preprocess_scm(model, subset, aggregate=True, df=None):
    warm_cols = ['cossim_warm', 'cossim_trustworthy', 'cossim_friendly', 'cossim_honest',
                 'cossim_likable', 'cossim_sincere']
    comp_cols = ['cossim_competent', 'cossim_intelligent', 'cossim_skilled', 'cossim_efficient',
                 'cossim_assertive', 'cossim_confident']

    if df is None:
        df = _load_causalface_data(model, subset, "scm")
    if aggregate:
        df["warm"] = df[warm_cols].mean(axis=1)
        df["comp"] = df[comp_cols].mean(axis=1)
        df.drop(columns=warm_cols + comp_cols, inplace=True)
    return df

def load_and_preprocess_abc(model, subset, aggregate=True, df=None):
    agency_pos = ["powerful", "high-status", "dominating", "wealthy", "confident", "competitive"]
    agency_neg = ["powerless", "low-status", "dominated", "poor", "meek", "passive"]
    belief_pos = ["science-oriented", "alternative", "liberal", "modern"]
    belief_neg = ["religious", "conventional", "conservative", "traditional"]
    communion_pos = ["trustworthy", "sincere", "friendly", "benevolent", "likable", "altruistic"]
    communion_neg = ["untrustworthy", "dishonest", "unfriendly", "threatening", "unpleasant", "egoistic"]
    all_colnames = [f"cossim_{c}" for c in
                    agency_pos + agency_neg + belief_pos + belief_neg + communion_pos + communion_neg]

    if df is None:
        df = _load_causalface_data(model, subset, "abc")

    if aggregate:
        df["a+"] = df[[f"cossim_{x}" for x in agency_pos]].mean(axis=1)
        df["a-"] = df[[f"cossim_{x}" for x in agency_neg]].mean(axis=1)
        df["b+"] = df[[f"cossim_{x}" for x in belief_pos]].mean(axis=1)
        df["b-"] = df[[f"cossim_{x}" for x in belief_neg]].mean(axis=1)
        df["c+"] = df[[f"cossim_{x}" for x in communion_pos]].mean(axis=1)
        df["c-"] = df[[f"cossim_{x}" for x in communion_neg]].mean(axis=1)

        df.drop(columns=all_colnames, inplace=True)
    return df


def load_all_causalface(model="clip", control=True, scm=True, abc=True, subsets=None, cossim_prefix="cossim_", aggregate=True):
    if aggregate:
        cossim_prefix=None

    if subsets is None:
        subsets = ["age", "lighting", "pose", "smiling"]
    elif isinstance(subsets, str):
        subsets = [subsets]

    outer_list = []
    for subset in subsets:
        dfs = []
        if control:
            dfs += [load_and_preprocess_control(model, subset).set_index(INDEX_COLS + [subset])]
        if scm:
            dfs += [load_and_preprocess_scm(model, subset, aggregate=aggregate).set_index(INDEX_COLS + [subset])]
        if abc:
            dfs += [load_and_preprocess_abc(model, subset, aggregate=aggregate).set_index(INDEX_COLS + [subset])]
        outer_list += [pd.concat(dfs, axis=1).reset_index()]

    df = pd.concat(outer_list).reset_index(drop=True)
    if cossim_prefix:
        df.rename(columns={c: cossim_prefix + c for c in df.columns if c in ATTRIBUTE_COLS}, inplace=True)
    return df
