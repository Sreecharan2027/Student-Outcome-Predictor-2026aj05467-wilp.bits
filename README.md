# Student Outcome Predictor: Classification and Streamlit App
- **Live app:** https://student-outcome-predictor-2026aj05467-wilpbits-9veup8x2gdrq6p9.streamlit.app


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

## GitHub repository
https://github.com/Sreecharan2027/Student-Outcome-Predictor-2026aj05467-wilp.bits

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
├── train_models.py
├── requirements.txt
├── README.md
├── test_data.csv
├── results.csv
└── data.csv
```

## Models used

All five models were trained on the same split and evaluated on the held-out test set
(885 rows). Precision, Recall and F1 are macro-averaged, AUC is one-vs-rest macro, and
MCC is the multi-class version.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7966 | 0.9005 | 0.7667 | 0.7198 | 0.7347 | 0.6635 |
| Decision Tree | 0.7751 | 0.8954 | 0.7362 | 0.7102 | 0.7201 | 0.6291 |
| KNN | 0.7514 | 0.8890 | 0.7326 | 0.6608 | 0.6790 | 0.5879 |
| Naive Bayes | 0.4452 | 0.7729 | 0.5902 | 0.5369 | 0.4555 | 0.3110 |
| Support Vector Machine | 0.8169 | 0.9022 | 0.8044 | 0.7291 | 0.7483 | 0.6995 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong and consistent (accuracy 0.7966, AUC 0.9005, MCC 0.6635), second only to SVM. Recall is 0.94 for Graduate and 0.77 for Dropout but only 0.45 for Enrolled. It is very fast and easy to interpret. |
| Decision Tree | Accuracy 0.7751 and AUC 0.8954 are close to Logistic Regression, but MCC (0.6291) is lower. A single tree is high-variance, yet it finds Enrolled students slightly better than the other models (recall 0.50). |
| KNN | Weakest of the working models (MCC 0.5879). It finds only 37% of Enrolled students and often labels them Graduate (76 of 159). Distances are less informative in the high-dimensional one-hot space, and results depend on k. |
| Naive Bayes | Accuracy 0.4452 is below the 0.50 that always guessing Graduate would give, yet AUC is 0.7729. It over-predicts Enrolled (recall 0.87 but precision 0.25) and labels 266 of 442 Graduates as Enrolled. Its independence and Gaussian assumptions are violated by one-hot columns and correlated features such as 1st and 2nd semester units. |
| Support Vector Machine | Best on every metric (accuracy 0.8169, AUC 0.9022, precision 0.8044, MCC 0.6995). It is excellent on Graduate (recall 0.97) and strong on Dropout (0.80), but Enrolled recall is only 0.42. Prediction is much slower than Logistic Regression. |

### Overall winner: Support Vector Machine

SVM is best on all six metrics, including MCC (0.6995) and macro F1 (0.7483), which
matter more than accuracy because the classes are imbalanced. Its margin over Logistic
Regression is modest (about 2 points of accuracy), so Logistic Regression is a good
choice if speed and interpretability matter more.

**Common weakness:** every model struggles with the **Enrolled** class. Enrolled
students are still in progress, so their outcome is less clear-cut, and most models
confuse them with Graduates. This pulls macro recall below macro precision for all
models except Naive Bayes.

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