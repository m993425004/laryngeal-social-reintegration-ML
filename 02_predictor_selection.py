"""
02_predictor_selection.py

Predictor stability assessment and multicollinearity evaluation.

Workflow:
    Development cohort only
            |
            v
    Bootstrap resampling (n=1000)
            |
            v
    LASSO logistic regression
            |
            v
    Predictor selection frequency
            |
            v
    Stable predictors (frequency >=70%)
            |
            v
    Variance Inflation Factor (VIF)

Notes:
    - No patient-level information included.
    - No simulated data.
    - No file paths hard-coded.
    - Analysis should be performed only within development cohort.
"""


import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample

from statsmodels.stats.outliers_influence import variance_inflation_factor



def bootstrap_lasso_selection(
    X,
    y,
    n_bootstrap=1000,
    selection_threshold=0.70,
    random_state=None
):
    """
    Assess predictor selection stability using bootstrap LASSO logistic regression.

    Parameters
    ----------
    X : pandas.DataFrame
        Predictor matrix from development cohort.

    y : pandas.Series
        Binary outcome.

    n_bootstrap : int
        Number of bootstrap samples.

    selection_threshold : float
        Minimum selection frequency for stable predictors.

    random_state : int, optional


    Returns
    -------
    selection_frequency : pandas.DataFrame
        Predictor selection frequency.

    stable_predictors : list
        Predictors selected in >= threshold proportion.
    """

    rng = np.random.RandomState(random_state)

    predictors = X.columns

    selection_count = pd.Series(
        0,
        index=predictors,
        dtype=float
    )


    for i in range(n_bootstrap):

        X_boot, y_boot = resample(
            X,
            y,
            replace=True,
            random_state=rng.randint(0, 100000)
        )


        lasso_model = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            max_iter=5000
        )


        lasso_model.fit(
            X_boot,
            y_boot
        )


        selected = predictors[
            np.abs(lasso_model.coef_[0]) > 1e-8
        ]


        selection_count.loc[selected] += 1


    selection_frequency = (
        selection_count / n_bootstrap
    ).sort_values(
        ascending=False
    )


    selection_frequency = (
        selection_frequency
        .reset_index()
    )

    selection_frequency.columns = [
        "Predictor",
        "Selection_frequency"
    ]


    stable_predictors = (
        selection_frequency
        .loc[
            selection_frequency["Selection_frequency"]
            >= selection_threshold,
            "Predictor"
        ]
        .tolist()
    )


    return (
        selection_frequency,
        stable_predictors
    )



def calculate_vif(
    X
):
    """
    Calculate variance inflation factor.

    Parameters
    ----------
    X : pandas.DataFrame
        Predictor matrix.

    Returns
    -------
    vif_table : pandas.DataFrame
    """


    vif_values = []


    for i in range(X.shape[1]):

        vif_values.append(
            variance_inflation_factor(
                X.values,
                i
            )
        )


    vif_table = pd.DataFrame(
        {
            "Predictor": X.columns,
            "VIF": vif_values
        }
    )


    return vif_table



def remove_high_collinearity(
    X,
    vif_threshold=5
):
    """
    Identify predictors with substantial multicollinearity.

    Note:
        Final variable retention should consider
        statistical properties and clinical relevance.

    """

    vif_table = calculate_vif(X)


    high_vif_predictors = (
        vif_table
        .loc[
            vif_table["VIF"] > vif_threshold,
            "Predictor"
        ]
        .tolist()
    )


    return (
        vif_table,
        high_vif_predictors
    )



if __name__ == "__main__":

    print(
        """
        Predictor selection module.

        Required inputs:
        - Development cohort predictor matrix (X)
        - Development cohort outcome vector (y)

        Procedure:
        1. Bootstrap LASSO selection
        2. Calculate selection frequency
        3. Identify stable predictors
        4. Evaluate multicollinearity using VIF
        """
    )