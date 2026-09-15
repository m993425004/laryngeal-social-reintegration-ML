01_feature_encoding.py

Feature encoding pipeline for machine learning prediction model.

Purpose:
    Encode clinical and patient-reported predictors before model training.

Notes:
    - No patient-level information is included.
    - No simulated data generation.
    - File paths are intentionally not hard-coded.
    - Variable definitions should be provided separately.

Workflow:
    Raw anonymized dataset
            |
            v
    Variable identification
            |
            v
    Categorical encoding
            |
            v
    Model-ready feature matrix

"""

import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def identify_feature_types(
    data: pd.DataFrame,
    categorical_features: list,
    continuous_features: list
):
    """
    Identify categorical and continuous predictors.

    Parameters
    ----------
    data : pandas.DataFrame
        Input anonymized dataset.

    categorical_features : list
        Names of categorical predictors.

    continuous_features : list
        Names of continuous predictors.

    Returns
    -------
    X_cat, X_cont
    """

    X_cat = data[categorical_features].copy()

    X_cont = data[continuous_features].copy()

    return X_cat, X_cont



def build_encoder(
    categorical_features: list,
    continuous_features: list
):
    """
    Construct preprocessing pipeline.

    Categorical variables:
        One-hot encoding

    Continuous variables:
        Passed through without transformation

    """

    encoder = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                categorical_features
            ),
            (
                "continuous",
                "passthrough",
                continuous_features
            )
        ]
    )

    return encoder



def encode_features(
    data: pd.DataFrame,
    categorical_features: list,
    continuous_features: list
):
    """
    Encode predictors.

    Returns
    -------
    encoded_features : pandas.DataFrame
        Feature matrix for model development.

    encoder :
        fitted preprocessing object
    """

    encoder = build_encoder(
        categorical_features,
        continuous_features
    )

    encoded_matrix = encoder.fit_transform(data)

    feature_names = encoder.get_feature_names_out()

    encoded_features = pd.DataFrame(
        encoded_matrix,
        columns=feature_names,
        index=data.index
    )

    return encoded_features, encoder



if __name__ == "__main__":

    """
    Example workflow:

    1. Load anonymized dataset
    2. Define variable categories
    3. Encode predictors

    Actual dataset loading and variable names
    should be configured by the user.
    """

    print(
        "Feature encoding module loaded. "
        "Provide anonymized dataset and variable definitions "
        "before execution."
    )
