# Student Outcome Predictor: Classification and Streamlit App
- **Live app:** https://student-outcome-predictor-2026aj05467-wilpbits-9veup8x2gdrq6p9.streamlit.app

## GitHub repository
https://github.com/Sreecharan2027/Student-Outcome-Predictor-2026aj05467-wilp.bits

## Problem statement
Universities want to spot students who are at risk of leaving before it is too late.
This project predicts whether a student will **Dropout**, stay **Enrolled**, or
**Graduate** by the end of the normal course duration, using background, family,
finance, admission and first/second-semester performance data.

This is a **3-class classification** problem. Five classifiers are trained on the
same data and compared with six metrics, and the results can be explored in an
interactive Streamlit app.

## Dataset description
- **Source:** UCI Machine Learning Repository, "Predict Students' Dropout and Academic Success"
- **Link:** https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success
- **Size:** 4,424 students, 36 input features, 1 target column
- **Target classes:**

| Class | distribution |
|---|---|
| Graduate | 2209 |
| Dropout | 1421 |
| Enrolled | 794 |

- **Test dataset:**  test_data.csv

- **Feature groups:** demographic (age, gender, marital status), socio-economic
  (parents' qualification and occupation, scholarship, debtor, tuition status),
  application details (mode, order, course, previous qualification), first and second
  semester academic results, and economic indicators (unemployment, inflation, GDP)
- **Missing values:** none
- **Class imbalance:** Enrolled is the smallest class, so MCC and macro-averaged metrics
  are reported alongside accuracy
- **Preprocessing:** one-hot encoding of nominal code columns (rare categories grouped),
  standard scaling of numeric columns, binary flags passed through. Preprocessing is
  fitted on training data only and saved inside each model pipeline
- **Train/test split:** test-0.2, stratified on Target, random_state = 4091;
  3539 training rows and 885 test rows (`test_data.csv`)

**File structure:**
``` text
Student-Outcome-Predictor-2026aj05467-wilp.bits/
├── .streamlit/
│   └── config.toml
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   ├── support_vector_machine.joblib
│   └── label_encoder.joblib
├── .gitignore
├── app.py
├── train_models.ipynb
├── requirements.txt
├── README.md
├── test_data.csv
├── results.csv
└── data.csv
```
### Models used

All five models were trained on the same split and evaluated on the held-out test set
(885 rows). Precision, Recall and F1 are macro-averaged, AUC is one-vs-rest macro, and
MCC is the multi-class version.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8011 | 0.9106 | 0.7676 | 0.7230 | 0.7360 | 0.6708 |
| Decision Tree | 0.7605 | 0.8479 | 0.7085 | 0.6806 | 0.6899 | 0.6028 |
| KNN | 0.7130 | 0.8254 | 0.6585 | 0.6084 | 0.6172 | 0.5194 |
| Naive Bayes | 0.4294 | 0.7466 | 0.5793 | 0.5061 | 0.4391 | 0.2822 |
| Support Vector Machine | 0.7932 | 0.8686 | 0.7580 | 0.6958 | 0.7077 | 0.6581 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best on all six metrics (accuracy 0.8011, AUC 0.9106, MCC 0.6708). It finds Graduates (recall 0.93) and Dropouts (0.80) well, but only 43% of Enrolled students. It is also the fastest to train and easy to interpret |
| Decision Tree | Behind Logistic Regression (accuracy 0.7605, AUC 0.8479, MCC 0.6028). A single depth-limited tree is high-variance and has to split on many sparse one-hot columns. |
| KNN | Weakest of the working models (accuracy 0.7130, MCC 0.5194). Distances become less informative in the high-dimensional one-hot space, and results depend on k.|
| Naive Bayes | Accuracy 0.4294 is below the 0.50 that always guessing Graduate would give, yet AUC is 0.7466, so it ranks classes better than its hard labels show. Its independence and Gaussian assumptions are violated by one-hot columns and by correlated features such as 1st and 2nd semester units. |
| Support Vector Machine | Close to Logistic Regression on accuracy (0.7932) and precision (0.7580), but lower on AUC (0.8686), recall (0.6958) and MCC (0.6581). |

### Overall winner: Logistic Regression

Logistic Regression is best on all six metrics, including MCC (0.6708) and macro F1
(0.7360), which matter more than accuracy because the classes are imbalanced
(Graduate 49.9%, Dropout 32.1%, Enrolled 17.9%). SVM is a close second on accuracy, but
Logistic Regression also trains and predicts faster and is easier to interpret.

**Common weakness:** every model struggles with the **Enrolled** class. For Logistic
Regression, Enrolled recall is only 0.43 (precision 0.66), compared with 0.93 for
Graduate and 0.80 for Dropout. Enrolled students are still in progress, so their outcome
is less clear-cut. This pulls macro recall below macro precision for all five models.

## Streamlit app
- **Live app:** https://student-outcome-predictor-2026aj05467-wilpbits-9veup8x2gdrq6p9.streamlit.app
- **Features:** CSV upload of test data, model dropdown, six evaluation metrics,
  confusion matrix and classification report, and a comparison table of all five models
- Models are loaded from `model/` and are **not retrained** on uploaded data

## f. How to run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
To retrain the models: `python train_models.py`
