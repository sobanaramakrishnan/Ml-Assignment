# Dry Bean Classification: Multi-Model Comparison and Streamlit Deployment

Machine Learning, Assignment 2 | M.Tech (AIML), BITS Pilani WILP

## a. Problem Statement

Dry beans are sold in several varieties, and sorting them by hand is slow and inconsistent. The goal is to predict the **variety of a dry bean** (7 classes: Barbunya, Bombay, Cali, Dermason, Horoz, Seker, Sira) from 16 shape and size measurements taken from images of each bean. This is a **multi-class classification** problem. Reliable automated classification supports quality control and seed certification in agriculture. Five classifiers (Logistic Regression, Decision Tree, KNN, Naive Bayes, SVM) are trained on the same data, compared on six metrics, and served through a Streamlit app.

## b. Dataset Description

- **Source:** UCI Machine Learning Repository, _Dry Bean Dataset_ (Koklu & Ozkan, 2020), <https://archive.ics.uci.edu/dataset/602/dry+bean+dataset>
- **Instances:** 13,611 raw rows; about 13,543 after duplicate removal (the 2,709-row test set is 20% of the de-duplicated data) | **Features:** 16 | **Target:** `Class` (7 classes)
- **Feature types:** all numeric. Size features: `Area`, `Perimeter`, `MajorAxisLength`, `MinorAxisLength`, `ConvexArea`, `EquivDiameter`. Shape features: `AspectRation`, `Eccentricity`, `Extent`, `Solidity`, `roundness`, `Compactness`, `ShapeFactor1` to `ShapeFactor4`. (`AspectRation` is spelled this way in the original dataset.)
- **Class balance (test set, 2,709 rows):** DERMASON 709, SIRA 527, SEKER 406, HOROZ 372, CALI 326, BARBUNYA 265, BOMBAY 104. The classes are imbalanced, with BOMBAY the smallest.
- **Preprocessing:** duplicate removal, median imputation and standard scaling for numeric features (there are no missing values and no categorical features, so the one-hot step is a no-op), label encoding of the target, stratified 80/20 train/test split (`random_state=42`). All preprocessing sits inside a scikit-learn `Pipeline`, so scaling is fitted on training data only (no leakage).
- **Metric averaging (multi-class):** Precision, Recall and F1 are **weighted** averages; AUC is **one-vs-rest macro**. Weighted Recall is mathematically equal to Accuracy.

## c. GitHub Repository

https://github.com/sobanaramakrishnan/Ml-Assignment.git

```
├── app.py              # Streamlit app
├── train_models.py     # training + evaluation
├── ml_utils.py         # shared metric code
├── requirements.txt
├── test_data.csv       # held-out test set (2,709 rows)
├── data/dataset.csv    # full Dry Bean data (needed only to re-run training)
└── model/              # 5 saved .joblib pipelines, meta.json, metrics_comparison.csv
```

Live app: https://ml-assignment-jxtzhfw9wnmhrytygpejek.streamlit.app/

## d. Models Used

### Comparison Table (test set, 2,709 rows)

| ML Model Name          | Accuracy |    AUC | Precision | Recall |     F1 |    MCC |
| :--------------------- | -------: | -----: | --------: | -----: | -----: | -----: |
| Logistic Regression    |   0.9192 | 0.9947 |    0.9197 | 0.9192 | 0.9193 | 0.9023 |
| Decision Tree          |   0.8837 | 0.9807 |    0.8854 | 0.8837 | 0.8839 | 0.8595 |
| KNN                    |   0.9136 | 0.9854 |    0.9144 | 0.9136 | 0.9138 | 0.8956 |
| Naive Bayes            |   0.8970 | 0.9911 |    0.8997 | 0.8970 | 0.8972 | 0.8762 |
| Support Vector Machine |   0.9210 | 0.9945 |    0.9214 | 0.9210 | 0.9211 | 0.9045 |

### Observations

| ML Model Name          | Observation about model performance                                                                                                                                                                                                                                                                                                                                                       |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Logistic Regression    | A strong linear baseline on standardised features: accuracy 0.9192, F1 0.9193, MCC 0.9023, and the highest AUC (0.9947). Many shape features are highly correlated (for example `Area`, `ConvexArea`, `EquivDiameter`), which scaling and L2 regularisation handle well. It trails the SVM by only about 0.002 in accuracy, which shows that the classes are close to linearly separable. |
| Decision Tree          | The weakest model on every metric: accuracy 0.8837, MCC 0.8595, AUC 0.9807. Depth is capped at 6 with `min_samples_leaf=5` to limit overfitting, but axis-aligned splits struggle with the smooth, overlapping boundaries between similar bean types. Its advantage is interpretability.                                                                                                  |
| KNN                    | Good performance after scaling: accuracy 0.9136, MCC 0.8956, though its AUC (0.9854) is lower than that of LR, SVM and Naive Bayes. Distance-weighted voting with k=7 gives coarser class probabilities than the other models. It is sensitive to feature scaling and slower at prediction time because it stores the whole training set.                                                 |
| Naive Bayes            | Accuracy 0.8970 and MCC 0.8762 are below the top three models, because the Gaussian independence assumption is violated by strongly correlated size features. Its AUC is still high (0.9911), which means it ranks the classes well even when its hard predictions are less accurate. It is simple and very fast to train.                                                                |
| Support Vector Machine | The best model on Accuracy (0.9210), Precision (0.9214), F1 (0.9211) and MCC (0.9045); its AUC (0.9945) is essentially tied with Logistic Regression. The RBF kernel captures non-linear class boundaries. The cost is slower training, and `probability=True` adds an internal cross-validation step.                                                                                    |
| **Overall winner**     | **Support Vector Machine**, with the highest MCC (0.9045) and F1 (0.9211). MCC is the main criterion because it stays reliable on imbalanced multi-class data. The margin over Logistic Regression (MCC 0.9023) is very small, so Logistic Regression is a simpler and faster alternative with almost the same quality.                                                                   |

## Streamlit App Features

- Upload a test CSV (it must contain the 16 feature columns and `Class`), or use the bundled `test_data.csv`
- Model selection dropdown (5 models)
- Evaluation metrics: Accuracy, AUC, Precision, Recall, F1, MCC
- Confusion matrix and classification report
- "Compare all models" tab with a metrics table and chart
- Saved models are loaded from `model/`; nothing is retrained on the uploaded data

## How to run

```bash
pip install -r requirements.txt
# put the UCI Dry Bean data (Class column included) at data/dataset.csv
python train_models.py
streamlit run app.py
```
