"""
09_sensitivity_analysis_SDSS.py

Sensitivity analysis using alternative SDSS cutoff.

Primary outcome:
    Poor social reintegration defined as SDSS >= 2

Sensitivity outcome:
    Poor social reintegration defined as SDSS >= 3


Purpose:
    Evaluate robustness of the prediction model
    under an alternative outcome definition.

Important:
    - Predictor variables remain unchanged.
    - Modeling framework remains unchanged.
    - Validation remains independent.
    - No patient-level information included.

"""


import numpy as np
import pandas as pd



from sklearn.metrics import (

    roc_auc_score,

    accuracy_score,

    recall_score,

    precision_score,

    f1_score,

    confusion_matrix,

    brier_score_loss

)



# ==========================================================
# 1. Generate alternative SDSS outcome
# ==========================================================


def create_sdss_outcome(

        sdss_score,

        cutoff=3

):

    """
    Create binary outcome using alternative SDSS cutoff.

    Parameters
    ----------
    sdss_score:
        SDSS total score

    cutoff:
        Threshold for poor social reintegration


    Returns
    -------
    binary outcome

    """


    outcome = (

        sdss_score >= cutoff

    ).astype(int)


    return outcome




# ==========================================================
# 2. Evaluate sensitivity model
# ==========================================================


def evaluate_sensitivity_model(

        model,

        X_validation,

        y_validation,

        threshold

):

    """
    Evaluate model performance under
    alternative outcome definition.

    Model must be trained using
    sensitivity-analysis outcome.

    """


    probability = model.predict_proba(

        X_validation

    )[:,1]



    prediction = (

        probability >= threshold

    ).astype(int)



    tn, fp, fn, tp = confusion_matrix(

        y_validation,

        prediction

    ).ravel()



    results = {


        "AUC":

            roc_auc_score(

                y_validation,

                probability

            ),


        "Accuracy":

            accuracy_score(

                y_validation,

                prediction

            ),


        "Sensitivity":

            recall_score(

                y_validation,

                prediction

            ),


        "Specificity":

            tn/(tn+fp),


        "PPV":

            precision_score(

                y_validation,

                prediction,

                zero_division=0

            ),


        "F1_score":

            f1_score(

                y_validation,

                prediction

            ),


        "Brier_score":

            brier_score_loss(

                y_validation,

                probability

            )

    }


    return pd.DataFrame(

        [results]

    )




# ==========================================================
# 3. Compare primary and sensitivity outcomes
# ==========================================================


def compare_outcome_definitions(

        primary_results,

        sensitivity_results

):

    """
    Compare model performance under
    different SDSS definitions.

    """



    comparison = pd.concat(

        [

            primary_results.assign(

                Outcome_definition="SDSS>=2"

            ),


            sensitivity_results.assign(

                Outcome_definition="SDSS>=3"

            )

        ],

        ignore_index=True

    )


    return comparison




if __name__ == "__main__":


    print(

        """

        SDSS sensitivity analysis module.


        Primary definition:
            SDSS >= 2


        Sensitivity definition:
            SDSS >= 3


        Purpose:
            Evaluate robustness of findings
            to alternative outcome threshold.


        """

    )