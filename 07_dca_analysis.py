"""
07_dca_analysis.py

Decision Curve Analysis (DCA)

Purpose:
    Evaluate clinical utility of prediction models
    by calculating net benefit across a range of
    threshold probabilities.

Important:
    - DCA is used for clinical utility evaluation.
    - DCA is not used to select the classification threshold.
    - Models are evaluated using predicted probabilities only.

"""



import numpy as np
import pandas as pd




# ==========================================================
# 1. Calculate net benefit
# ==========================================================


def calculate_net_benefit(
        y_true,
        predicted_probability,
        threshold_probability
):

    """
    Calculate net benefit at a specific threshold.

    Formula:

    Net Benefit =
    TP/N - FP/N * (threshold/(1-threshold))


    Parameters
    ----------
    y_true:
        Observed outcome

    predicted_probability:
        Model predicted probability

    threshold_probability:
        Clinical decision threshold


    Returns
    -------
    net_benefit

    """



    predicted_positive = (

        predicted_probability

        >=

        threshold_probability

    )


    true_positive = np.sum(

        predicted_positive

        &

        (y_true == 1)

    )


    false_positive = np.sum(

        predicted_positive

        &

        (y_true == 0)

    )


    n = len(y_true)



    if threshold_probability == 1:

        return 0



    net_benefit = (

        true_positive / n

        -

        (

            false_positive / n

        )

        *

        (

            threshold_probability

            /

            (1-threshold_probability)

        )

    )


    return net_benefit




# ==========================================================
# 2. Treat-all strategy
# ==========================================================


def calculate_treat_all_net_benefit(

        y_true,

        threshold_probability

):

    """
    Net benefit assuming all patients
    receive intervention.

    """


    prevalence = np.mean(

        y_true

    )


    net_benefit = (

        prevalence

        -

        (

            1-prevalence

        )

        *

        (

            threshold_probability

            /

            (1-threshold_probability)

        )

    )


    return net_benefit




# ==========================================================
# 3. Treat-none strategy
# ==========================================================


def calculate_treat_none_net_benefit():

    """
    Treat-none strategy always has zero benefit.
    """


    return 0




# ==========================================================
# 4. Generate DCA curve
# ==========================================================


def generate_dca_curve(

        y_true,

        prediction_dict,

        thresholds=None

):

    """
    Generate decision curve analysis results.

    Parameters
    ----------
    y_true:
        Outcome vector


    prediction_dict:
        Dictionary containing model predictions

        Example:

        {
            "LR": probabilities,
            "RF": probabilities
        }


    thresholds:
        Clinical threshold probability range


    Returns
    -------
    DCA dataframe

    """


    if thresholds is None:

        thresholds = np.arange(

            0.01,

            0.99,

            0.01

        )



    results = []



    for threshold in thresholds:


        row = {


            "threshold_probability":

                threshold,


            "treat_none":

                calculate_treat_none_net_benefit(),


            "treat_all":

                calculate_treat_all_net_benefit(

                    y_true,

                    threshold

                )

        }



        for model_name, probability in prediction_dict.items():


            row[model_name] = calculate_net_benefit(

                y_true,

                probability,

                threshold

            )


        results.append(row)



    return pd.DataFrame(results)




# ==========================================================
# 5. Clinical utility comparison
# ==========================================================


def summarize_dca(

        dca_results,

        threshold_range=(0.05,0.50)

):

    """
    Summarize net benefit within
    clinically relevant thresholds.

    """



    filtered = dca_results.loc[

        (

            dca_results[

                "threshold_probability"

            ]

            >=

            threshold_range[0]

        )

        &

        (

            dca_results[

                "threshold_probability"

            ]

            <=

            threshold_range[1]

        )

    ]


    summary = (

        filtered

        .describe()

    )


    return summary




if __name__ == "__main__":


    print(

        """

        Decision Curve Analysis module.

        Inputs:

            - observed outcomes
            - predicted probabilities
            - candidate models


        Outputs:

            - net benefit curves
            - treat-all strategy
            - treat-none strategy


        Note:

        DCA evaluates clinical utility and
        does not determine the final cutoff.


        """

    )