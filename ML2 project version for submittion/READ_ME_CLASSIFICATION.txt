
================================================================================
PART 2: CLASSIFICATION PROJECT – CREDIT CARD DEFAULT PREDICTION
================================================================================

1. PROBLEM DESCRIPTION
----------------------
Goal:
Predict whether a credit card client will default on their payment in the next
month based on demographic information, credit limits, billing history, and
repayment behavior using supervised classification models.

Target Variable:
- default payment next month (binary: 0 = No default, 1 = Default)

Dataset:
- Default of Credit Card Clients Dataset

The problem is formulated as a binary classification task with emphasis on
model comparison, interpretability, and correct evaluation under class imbalance.

-------------------------------------------------------------------------------

2. FILES INCLUDED
-----------------
- classification_credit_default.ipynb : Main Jupyter Notebook containing data
                                        exploration, preprocessing, modeling,
                                        tuning, and evaluation.
- classification_output.pdf            : Exported outputs, tables, and figures.
- default of credit card clients.csv   : Dataset used for training and evaluation.
- README.txt                           : Project description and reproducibility
                                        instructions.

-------------------------------------------------------------------------------

3. DATA SOURCE & CONTENT
------------------------
Source:
- UCI Machine Learning Repository:
  “Default of Credit Card Clients Dataset”

Dataset Characteristics:
- Observations: 30,000
- Features: 24 input variables + 1 target variable
- Target variable: default payment next month

Feature Groups:
- Demographic variables:
  sex, age, education, marriage
- Credit profile:
  credit limit (LIMIT_BAL)
- Repayment status variables:
  PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6
- Billing amounts:
  BILL_AMT1 – BILL_AMT6
- Payment amounts:
  PAY_AMT1 – PAY_AMT6

Repayment status variables encode recent payment delays and are expected to be
the strongest predictors of default.

-------------------------------------------------------------------------------

4. EXPERIMENTAL SETUP (METHODOLOGY)
----------------------------------

A. Data Preprocessing
---------------------
- Identifier Handling:
  The ID column was excluded from modeling as it contains no predictive
  information.

- Missing Values:
  The dataset contains no missing values.

- Train–Test Split:
  Data was split into training and test sets using an 80/20 ratio with
  stratification on the target variable to preserve class proportions.

- Feature Scaling:
  Numerical features were standardized for Logistic Regression using
  StandardScaler within a Pipeline. Tree-based models were trained on raw values,
  as they are robust to feature scaling.

- Class Imbalance:
  The target variable is moderately imbalanced (~22% default rate). Evaluation
  therefore focused on precision, recall, and F1-score rather than accuracy
  alone.

----------------------------------

B. Models & Hyperparameter Tuning
---------------------------------
Hyperparameter tuning was conducted using GridSearchCV with cross-validation.
F1-score was selected as the primary optimization metric.

Models evaluated:

1. Logistic Regression (Baseline)
   - Used as a benchmark linear classifier
   - Combined with feature scaling

2. Random Forest Classifier (Ensemble – Bagging)
   - Tuned parameters:
     - n_estimators
     - max_depth
     - min_samples_split

3. Gradient Boosting Classifier (Ensemble – Boosting)
   - Tuned parameters:
     - n_estimators
     - learning_rate
     - max_depth

A fixed random_state was used across models to ensure reproducibility.

-------------------------------------------------------------------------------

5. MODEL EVALUATION
-------------------
Evaluation was performed on a held-out test set.

Metrics:
- Accuracy
- Precision
- Recall
- F1-score

Interpretation Note:
Due to class imbalance, recall and F1-score for the default class (1) were
considered the most important metrics.

Best Performing Model:
- Gradient Boosting Classifier

Test Set Performance (Gradient Boosting):
- Accuracy ≈ 0.82
- Precision (default class) ≈ 0.67
- Recall (default class) ≈ 0.36
- F1-score (default class) ≈ 0.47

-------------------------------------------------------------------------------

6. ERROR ANALYSIS
-----------------
Confusion matrix analysis indicates:
- High accuracy in identifying non-defaulting clients.
- A substantial number of false negatives remain, reflecting the inherent
  difficulty of detecting defaults under class imbalance.
- Repayment history variables (PAY_*) play a dominant role in distinguishing
  defaulters from non-defaulters.

This trade-off highlights the balance between minimizing financial risk and
avoiding excessive rejection of creditworthy clients.

-------------------------------------------------------------------------------

7. RESULTS & CONCLUSIONS
------------------------
Key findings:
1. Tree-based ensemble models outperform Logistic Regression, indicating
   non-linear relationships in the data.
2. Repayment status variables are the most informative predictors of default.
3. Gradient Boosting achieves the best balance between precision and recall for
   the default class.

Overall, Gradient Boosting was selected as the final model due to its superior
F1-score and stable performance.

-------------------------------------------------------------------------------

8. ETHICAL CONSIDERATIONS
-------------------------
Credit default prediction models may encode historical biases present in
financial data. False positives can unfairly restrict access to credit, while
false negatives may expose financial institutions to risk. These models should
be used as decision-support tools rather than fully automated systems, with
appropriate transparency, monitoring, and fairness considerations.

-------------------------------------------------------------------------------

9. REPRODUCIBILITY INSTRUCTIONS
-------------------------------
1. Ensure 'default of credit card clients.csv' is located in the same directory
   as the notebook.
2. Required libraries:
   - pandas
   - numpy
   - matplotlib
   - seaborn
   - scikit-learn
3. Python version: 3.10+
4. All models use fixed random_state values for reproducibility.
5. Run the Jupyter Notebook from top to bottom to reproduce all results.

-------------------------------------------------------------------------------




