"""
03_model_training.py

Machine learning model development pipeline.

Purpose:
    Train and optimize multiple machine learning algorithms
    using the development cohort only.

Algorithms:
    1. Logistic Regression (LR)
    2. Decision Tree (DT)
    3. Random Forest (RF)
    4. Support Vector Machine (SVM)
    5. Gradient Boosting (GB)
    6. XGBoost
    7. LightGBM
    8. CatBoost

Methodological principles:
    - No class resampling.
    - No synthetic data generation.
    - No validation cohort information leakage.
    - Hyperparameter optimization performed only within
      the development cohort.

"""

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.svm import SVC

from sklearn.model_selection import RandomizedSearchCV


from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier



# ==========================================================
# 1. Define candidate models
# ==========================================================


def define_models(random_state=42):

    """
    Define all candidate algorithms.

    Returns
    -------
    models : dict

    """


    models = {


        "LR":
        Pipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler()
                ),

                (
                    "model",
                    LogisticRegression(
                        max_iter=5000,
                        random_state=random_state
                    )
                )
            ]
        ),



        "DT":

        DecisionTreeClassifier(
            random_state=random_state
        ),



        "RF":

        RandomForestClassifier(
            random_state=random_state
        ),



        "SVM":

        Pipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler()
                ),

                (
                    "model",
                    SVC(
                        probability=True,
                        random_state=random_state
                    )
                )
            ]
        ),



        "GB":

        GradientBoostingClassifier(
            random_state=random_state
        ),



        "XGBoost":

        XGBClassifier(
            random_state=random_state,
            eval_metric="logloss"
        ),



        "LightGBM":

        LGBMClassifier(
            random_state=random_state,
            verbosity=-1
        ),



        "CatBoost":

        CatBoostClassifier(
            random_state=random_state,
            verbose=False
        )

    }


    return models




# ==========================================================
# 2. Hyperparameter search space
# ==========================================================


def define_parameter_spaces():


    """
    Predefined hyperparameter spaces.

    Randomized search is used because of
    limited clinical sample size.

    """


    parameter_spaces = {


        # Logistic regression
        "LR":

        {

            "model__C":

            [
                0.001,
                0.01,
                0.1,
                1,
                10,
                100
            ]

        },



        # Decision tree
        "DT":

        {

            "max_depth":

            [
                None,
                3,
                5,
                7,
                10
            ],


            "min_samples_split":

            [
                2,
                5,
                10
            ],


            "min_samples_leaf":

            [
                1,
                3,
                5,
                10
            ],


            "criterion":

            [
                "gini",
                "entropy"
            ]

        },



        # Random forest
        "RF":

        {

            "n_estimators":

            [
                200,
                500,
                800
            ],


            "max_depth":

            [
                None,
                5,
                10,
                20
            ],


            "min_samples_split":

            [
                2,
                5,
                10
            ],


            "min_samples_leaf":

            [
                1,
                2,
                5
            ],


            "max_features":

            [
                "sqrt",
                "log2"
            ]

        },



        # SVM
        "SVM":

        {

            "model__C":

            [
                0.01,
                0.1,
                1,
                10,
                100
            ],


            "model__gamma":

            [
                "scale",
                "auto"
            ]

        },



        # Gradient boosting
        "GB":

        {

            "n_estimators":

            [
                100,
                300,
                500
            ],


            "learning_rate":

            [
                0.01,
                0.05,
                0.1
            ],


            "max_depth":

            [
                2,
                3,
                5
            ],


            "subsample":

            [
                0.7,
                0.9,
                1.0
            ]

        },



        # XGBoost
        "XGBoost":

        {

            "n_estimators":

            [
                100,
                300,
                500
            ],


            "learning_rate":

            [
                0.01,
                0.05,
                0.1
            ],


            "max_depth":

            [
                2,
                3,
                5
            ],


            "subsample":

            [
                0.7,
                0.9,
                1.0
            ],


            "colsample_bytree":

            [
                0.7,
                0.9,
                1.0
            ],


            "reg_lambda":

            [
                1,
                5,
                10
            ]

        },



        # LightGBM
        "LightGBM":

        {

            "n_estimators":

            [
                100,
                300,
                500
            ],


            "learning_rate":

            [
                0.01,
                0.05,
                0.1
            ],


            "num_leaves":

            [
                15,
                31,
                63
            ],


            "max_depth":

            [
                -1,
                5,
                10
            ],


            "feature_fraction":

            [
                0.7,
                0.9,
                1.0
            ]

        },



        # CatBoost
        "CatBoost":

        {

            "iterations":

            [
                200,
                500
            ],


            "depth":

            [
                4,
                6,
                8
            ],


            "learning_rate":

            [
                0.01,
                0.05,
                0.1
            ],


            "l2_leaf_reg":

            [
                1,
                3,
                5
            ]

        }

    }


    return parameter_spaces




# ==========================================================
# 3. Train all candidate models
# ==========================================================


def train_candidate_models(
        X_development,
        y_development,
        cv,
        n_iter=100,
        scoring="roc_auc",
        random_state=42
):

    """
    Train and optimize all candidate models.

    Parameters
    ----------
    X_development:
        Development cohort predictors

    y_development:
        Development cohort outcome

    cv:
        Cross-validation strategy

    Returns
    -------
    trained_models

    """


    models = define_models(
        random_state=random_state
    )


    parameter_spaces = define_parameter_spaces()


    trained_models = {}


    for name, model in models.items():


        search = RandomizedSearchCV(

            estimator=model,

            param_distributions=
                parameter_spaces[name],

            n_iter=n_iter,

            scoring=scoring,

            cv=cv,

            random_state=random_state,

            n_jobs=-1

        )


        search.fit(

            X_development,

            y_development

        )


        trained_models[name] = {

            "model":
                search.best_estimator_,


            "best_parameters":
                search.best_params_,


            "best_cv_score":
                search.best_score_

        }


    return trained_models




# ==========================================================
# 4. Train final LR model
# ==========================================================


def train_final_lr_model(
        X_development,
        y_development,
        cv,
        random_state=42
):

    """
    Train final logistic regression model.

    Final model selection should be based on:
        - discrimination
        - calibration
        - clinical utility
        - interpretability

    """


    lr_pipeline = Pipeline(

        steps=[

            (
                "scaler",
                StandardScaler()
            ),


            (
                "model",

                LogisticRegression(

                    max_iter=5000,

                    random_state=random_state

                )

            )

        ]

    )


    search = RandomizedSearchCV(

        estimator=lr_pipeline,

        param_distributions={

            "model__C":

            [

                0.001,
                0.01,
                0.1,
                1,
                10,
                100

            ]

        },


        n_iter=6,

        scoring="roc_auc",

        cv=cv,

        random_state=random_state,

        n_jobs=-1

    )


    search.fit(

        X_development,

        y_development

    )


    return search.best_estimator_




if __name__ == "__main__":


    print(

        """
        Model training pipeline.

        Inputs:
            Development cohort only

        Outputs:
            Optimized candidate models
            Final logistic regression model

        Validation cohort is not used.
        No resampling is performed.
        """

    )