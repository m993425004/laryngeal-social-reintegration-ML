"""
05_threshold_optimization.py

Threshold optimization using Youden index.

Purpose:
    Determine the optimal classification threshold
    from the development cohort.

Important:
    - Threshold is determined ONLY from development data.
    - The derived threshold should be fixed before
      application to temporal validation cohort.
    - No validation information is used.

"""



import numpy as np
import pandas as pd


from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    confusion_matrix
)



# ==========================================================
# 1. Calculate Youden index
# ==========================================================


def calculate_youden_threshold(
        y_true,
        y_probability
):

    """
    Calculate optimal threshold using Youden index.

    Youden index:

        J = sensitivity + specificity - 1

    Parameters
    ----------
    y_true:
        Development cohort outcome

    y_probability:
        Predicted probability from development cohort


    Returns
    -------
    threshold_results

    """


    fpr, tpr, thresholds = roc_curve(

        y_true,

        y_probability

    )


    specificity = 1 - fpr


    youden_index = (

        tpr + specificity - 1

    )


    optimal_index = np.argmax(

        youden_index

    )


    optimal_threshold = thresholds[

        optimal_index

    ]


    results = {


        "optimal_threshold":

            optimal_threshold,


        "sensitivity":

            tpr[optimal_index],


        "specificity":

            specificity[optimal_index],


        "youden_index":

            youden_index[optimal_index]

    }


    return pd.DataFrame(

        [results]

    )




# ==========================================================
# 2. Apply predefined threshold
# ==========================================================


def evaluate_threshold(
        y_true,
        y_probability,
        threshold
):

    """
    Apply threshold selected from development cohort.

    Used for:
        - temporal validation evaluation

    """


    prediction = (

        y_probability >= threshold

    ).astype(int)



    tn, fp, fn, tp = confusion_matrix(

        y_true,

        prediction

    ).ravel()



    sensitivity = tp / (tp + fn)


    specificity = tn / (tn + fp)



    return pd.DataFrame(

        [

            {

                "threshold":

                    threshold,


                "sensitivity":

                    sensitivity,


                "specificity":

                    specificity

            }

        ]

    )




# ==========================================================
# 3. ROC data export
# ==========================================================


def export_roc_curve_data(

        y_true,

        y_probability

):

    """
    Export ROC curve coordinates.

    """

    fpr, tpr, thresholds = roc_curve(

        y_true,

        y_probability

    )


    auc = roc_auc_score(

        y_true,

        y_probability

    )


    roc_data = pd.DataFrame(

        {

            "false_positive_rate":

                fpr,


            "true_positive_rate":

                tpr,


            "threshold":

                thresholds

        }

    )


    return roc_data, auc




if __name__ == "__main__":


    print(

        """

        Threshold optimization module.

        Procedure:

        Development cohort
                |
                v
        Predicted probabilities
                |
                v
        Youden index optimization
                |
                v
        Fixed threshold applied
        to temporal validation cohort


        """

    )