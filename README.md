# Ml-Assignment

# <Your Project Title> – Multi-Model Classification & Streamlit Deployment

## a. Problem Statement
<2–4 sentences: what are you predicting, why it matters, binary or multi-class, and the real-world use of the prediction.>

## b. Dataset Description  
- **Source:** <Kaggle/UCI name + URL>
- **Instances:** <N> | **Features:** <F> | **Target:** `<col>` (<classes and class balance>)
- **Feature types:** <numeric / categorical, key columns>
- **Preprocessing:** duplicate removal, median/mode imputation, standard scaling, one-hot encoding, stratified 80/20 split (seed 42).

## c. GitHub Repository
<https://github.com/<user>/<repo>>

```
├── app.py              # Streamlit app
├── train_models.py     # training + evaluation
├── ml_utils.py         # shared metric code
├── requirements.txt
├── test_data.csv       # held-out test set
└── model/              # 5 saved .joblib pipelines, meta.json, metrics_comparison.csv
```
Live app: <https://<your-app>.streamlit.app>

## d. Models Used

### Comparison Table (test set)
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | | | | | | |
| Decision Tree | | | | | | |
| KNN | | | | | | |
| Naive Bayes | | | | | | |
| Support Vector Machine | | | | | | |

*(Paste the table printed by `python train_models.py`.)*

### Observations
| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | <baseline strength; linear separability; coefficients interpretable; effect of scaling> |
| Decision Tree | <overfitting vs depth; interpretability; variance; compare train vs test> |
| KNN | <sensitivity to k and scaling; curse of dimensionality; slow inference> |
| Naive Bayes | <independence assumption vs correlated features; fast; calibration/AUC vs accuracy> |
| Support Vector Machine | <RBF kernel capturing non-linearity; cost of training; best/worst metric> |
| **Overall winner** | <model + justification using MCC/F1/AUC, not accuracy alone; mention trade-offs> |

## How to run
```bash
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```
