class Control:
    name = "control"
    all_attributes = ["<blank>", "white", "black", "asian", "male", "female", "young", "old"]


class StereotypeContentModel:
    name = "scm"

    warm = ["warm", "trustworthy", "friendly", "honest", "likable", "sincere"]
    comp = ["competent", "intelligent", "skilled", "efficient", "assertive", "confident"]
    all_attributes = warm + comp


class ABCModel:
    name = "abc"

    agency_pos = ["powerful", "high-status", "dominating", "wealthy", "confident", "competitive"]
    agency_neg = ["powerless", "low-status", "dominated", "poor", "meek", "passive"]
    belief_pos = ["science-oriented", "alternative", "liberal", "modern"]
    belief_neg = ["religious", "conventional", "conservative", "traditional"]
    communion_pos = ["trustworthy", "sincere", "friendly", "benevolent", "likable", "altruistic"]
    communion_neg = ["untrustworthy", "dishonest", "unfriendly", "threatening", "unpleasant", "egoistic"]
    all_attributes = agency_pos + agency_neg + belief_pos + belief_neg + communion_pos + communion_neg


class RaceVariations:
    name = "race_synonyms"

    white_race_synonyms = ["White", "Caucasian", "European descent", "Euro-American", "Anglo", "Western European",
                           "European American", "White ethnic", "Non-Hispanic White"]

    asian_race_synonyms = ["Asian", "Oriental", "East Asian", "South Asian", "Southeast Asian",
                           "Asian American", "Asian-Pacific Islander", "Desi", "Far Eastern"]

    black_race_synonyms = ["Black","African American", "Afro-American", "African", "Afro-Caribbean",
                           "Negro", "Afro-Latinx", "Black African", "Sub-Saharan African"]

    all_attributes = white_race_synonyms + black_race_synonyms + asian_race_synonyms