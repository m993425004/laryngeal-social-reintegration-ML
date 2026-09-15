"""
06_calibration_analysis.py

Calibration assessment for clinical prediction model.

Purpose:
    Evaluate calibration performance of the final model.

Metrics:
    - Calibration curve
    - Calibration intercept
    - Calibration slope
    - Calibration-in-the-large (CITL)
    - Brier score
    - Hosmer-Lemeshow test

Important:
    - Model parameters are not updated.
    - Validation cohort is used only for evaluation.
    - No recalibration is performed.

"""



import numpy as np
import pandas as pd


from sklearn.metrics import (
    brier_score_loss
)


from sklearn.linear_model import LogisticRegression


from scipy.special import expit, logit


from scipy.stats import chi2



# ==========================================================
# 1. Calibration curve data
# ==========================================================


def calculate_calibration_curve(
        y_true,
        y_probability,
        n_bins=10
):

    """
    Generate calibration curve data.

    Parameters
    ----------
    y_true:
        Observed outcomes

    y_probability:
        Predicted probabilities

    n_bins:
        Number of probability groups


    Returns
    -------
    calibration_table

    """


    calibration_data = pd.DataFrame(

        {

            "observed":

                y_true.values,


            "predicted":

                y_probability

        }

    )


    calibration_data["bin"] = pd.qcut(

        calibration_data["predicted"],

        q=n_bins,

        duplicates="drop"

    )



    calibration_table = (

        calibration_data

        .groupby("bin")

        .agg(

            mean_predicted=(

                "predicted",

                "mean"

            ),


            mean_observed=(

                "observed",

                "mean"

            ),


            n=(

                "observed",

                "count"

            )

        )

        .reset_index()

    )


    return calibration_table




# ==========================================================
# 2. Calibration intercept and slope
# ==========================================================


def calculate_calibration_parameters(

        y_true,

        y_probability

):

    """
    Estimate calibration intercept and slope.

    Logistic calibration model:

        logit(Y)
        =
        intercept
        +
        slope * logit(predicted probability)

    """


    epsilon = 1e-10


    logit_probability = logit(

        np.clip(

            y_probability,

            epsilon,

            1-epsilon

        )

    )


    calibration_model = LogisticRegression(

        penalty=None,

        solver="lbfgs",

        max_iter=1000

    )


    calibration_model.fit(

        logit_probability.reshape(-1,1),

        y_true

    )


    intercept = (

        calibration_model.intercept_[0]

    )


    slope = (

        calibration_model.coef_[0][0]

    )


    return {

        "calibration_intercept":

            intercept,


        "calibration_slope":

            slope

    }




# ==========================================================
# 3. Calibration-in-the-large
# ==========================================================


def calculate_citl(

        y_true,

        y_probability

):

    """
    Calculate calibration-in-the-large.

    CITL represents systematic
    over- or under-prediction.

    """


    observed_prevalence = np.mean(

        y_true

    )


    predicted_mean = np.mean(

        y_probability

    )


    citl = logit(

        observed_prevalence

    ) - logit(

        predicted_mean

    )


    return citl




# ==========================================================
# 4. Hosmer-Lemeshow test
# ==========================================================


def hosmer_lemeshow_test(

        y_true,

        y_probability,

        groups=10

):

    """
    Hosmer-Lemeshow goodness-of-fit test.

    """



    data = pd.DataFrame(

        {

            "y":

                y_true.values,


            "probability":

                y_probability

        }

    )


    data["group"] = pd.qcut(

        data["probability"],

        q=groups,

        duplicates="drop"

    )


    grouped = (

        data

        .groupby("group")

        .agg(

            observed=(

                "y",

                "sum"

            ),


            expected=(

                "probability",

                "sum"

            ),


            total=(

                "y",

                "count"

            )

        )

    )


    chi_square = np.sum(

        (

            grouped["observed"]

            -

            grouped["expected"]

        )**2

        /

        (

            grouped["expected"]

            +

            1e-10

        )

        +

        (

            (

                grouped["total"]

                -

                grouped["observed"]

            )

            -

            (

                grouped["total"]

                -

                grouped["expected"]

            )

        )**2

        /

        (

            grouped["total"]

            -

            grouped["expected"]

            +

            1e-10

        )

    )


    degrees_of_freedom = groups - 2


    p_value = 1 - chi2.cdf(

        chi_square,

        degrees_of_freedom

    )


    return {

        "HL_chi_square":

            chi_square,


        "HL_p_value":

            p_value

    }




# ==========================================================
# 5. Complete calibration evaluation
# ==========================================================


def evaluate_calibration(

        y_true,

        y_probability

):

    """
    Complete calibration assessment.

    """


    calibration_parameters = calculate_calibration_parameters(

        y_true,

        y_probability

    )


    results = {


        "Brier_score":

            brier_score_loss(

                y_true,

                y_probability

            ),


        "CITL":

            calculate_citl(

                y_true,

                y_probability

            )

    }


    results.update(

        calibration_parameters

    )


    results.update(

        hosmer_lemeshow_test(

            y_true,

            y_probability

        )

    )


    return pd.DataFrame(

        [results]

    )




if __name__ == "__main__":


    print(

        """

        Calibration analysis module.

        Required inputs:

            - observed outcome
            - predicted probability


        Outputs:

            - calibration curve
            - calibration intercept
            - calibration slope
            - CITL
            - Brier score
            - Hosmer-Lemeshow test


        """

    )