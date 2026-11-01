import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from ml_utils import compute_metrics, METRIC_COLS

#Configuration
DATA_PATH   = "data/dataset.csv"   
TARGET_COL  = "Class"             
DROP_COLS   = []                  
TEST_SIZE   = 0.20
SEED        = 42

MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=SEED),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=5, random_state=SEED),
    "KNN": KNeighborsClassifier(n_neighbors=7, weights="distance"),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(kernel="rbf", C=1.0, probability=True, random_state=SEED),
}


def build_preprocessor(X):
    num = X.select_dtypes(include="number").columns.tolist()
    cat = [c for c in X.columns if c not in num]
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
    ])


def main():
    df = pd.read_csv(DATA_PATH).drop(columns=DROP_COLS).drop_duplicates()
    df = df.dropna(subset=[TARGET_COL])
    X, y_raw = df.drop(columns=[TARGET_COL]), df[TARGET_COL].astype(str)
    print(f"Instances: {len(df)} | Features: {X.shape[1]} | Classes: {y_raw.nunique()}")
    assert len(df) >= 500 and X.shape[1] >= 12, "Dataset violates minimum size rules!"

    le = LabelEncoder().fit(y_raw)
    y = le.transform(y_raw)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE,
                                              stratify=y, random_state=SEED)

    rows = {}
    for name, clf in MODELS.items():
        pipe = Pipeline([("prep", build_preprocessor(X)), ("clf", clf)]).fit(X_tr, y_tr)
        m = compute_metrics(y_te, pipe.predict(X_te), pipe.predict_proba(X_te))
        rows[name] = m
        fname = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(pipe, f"model/{fname}")
        print(f"{name:25s}", {k: round(v, 4) for k, v in m.items()})

    res = pd.DataFrame(rows).T[METRIC_COLS].round(4)
    res.index.name = "ML Model Name"
    res.to_csv("model/metrics_comparison.csv")
    print("\n", res.to_markdown())

    # artifacts used by the app
    test_out = X_te.copy()
    test_out[TARGET_COL] = le.inverse_transform(y_te)
    test_out.to_csv("test_data.csv", index=False)
    json.dump({"target_col": TARGET_COL, "classes": le.classes_.tolist(),
               "feature_cols": X.columns.tolist(),
               "models": {n: n.lower().replace(" ", "_") + ".joblib" for n in MODELS}},
              open("model/meta.json", "w"), indent=2)
    print("\nSaved models, meta.json, metrics_comparison.csv and test_data.csv")


if __name__ == "__main__":
    main()
