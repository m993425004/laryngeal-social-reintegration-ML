# laryngeal-social-reintegration-ML
Interpretable machine learning model for predicting poor social reintegration after laryngeal cancer surgery.
## Code Description


### Predictor processing

`01_feature_encoding.py`

Prepares clinical and patient-reported predictors for model development, including categorical variable encoding.


### Predictor selection

`02_predictor_selection.py`

Evaluates predictor stability using:

- 1,000 bootstrap samples
- LASSO logistic regression
- selection frequency calculation
- variance inflation factor assessment


Predictors selected in ≥70% of bootstrap samples are considered stable.


### Model development

`03_model_training.py`

Implements eight machine learning algorithms:

- Logistic regression
- Decision tree
- Random forest
- Support vector machine
- Gradient boosting
- XGBoost
- LightGBM
- CatBoost


Hyperparameter optimization is performed within the development cohort.

No class resampling procedures are applied.


### Temporal validation

`04_model_validation.py`

Evaluates the trained models in an independent temporal validation cohort.

No model fitting, feature selection, or parameter optimization is performed using validation data.


### Threshold optimization

`05_threshold_optimization.py`

Determines the optimal classification threshold using the Youden index in the development cohort.


### Calibration analysis

`06_calibration_analysis.py`

Evaluates:

- calibration curve
- calibration intercept
- calibration slope
- calibration-in-the-large
- Brier score
- Hosmer–Lemeshow test


### Decision curve analysis

`07_dca_analysis.py`

Evaluates clinical utility using net benefit across threshold probabilities.


### Model interpretation

`08_shap_interpretation.py`

Provides SHAP-based interpretation of model predictions.

SHAP values represent model attribution patterns and predictive associations rather than causal effects.


### Sensitivity analysis

`09_sensitivity_analysis_SDSS.py`

Evaluates model robustness using an alternative SDSS cutoff definition.


### Risk calculator

`10_risk_calculator.py`

Provides individualized risk estimation based on the final prediction model.

The calculator is intended as a decision-support prototype and should not replace clinical judgment.


---

## Data Availability


The original patient-level dataset is not publicly available because of ethical and privacy restrictions.

Only analytical code is provided in this repository.

Researchers requiring access to the data should contact the corresponding author and obtain appropriate institutional approval.


---

## Reproducibility


The analysis can be reproduced by:

1. Preparing an appropriately formatted anonymized dataset.
2. Running predictor preprocessing.
3. Performing predictor stability assessment.
4. Training candidate models within the development cohort.
5. Evaluating models using independent temporal validation data.
6. Performing calibration, decision curve analysis, SHAP interpretation, and sensitivity analysis.


---

## Software Environment


Python ≥3.9


Main packages:

- numpy
- pandas
- scikit-learn
- scipy
- statsmodels
- shap
- xgboost
- lightgbm
- catboost


Installation:
