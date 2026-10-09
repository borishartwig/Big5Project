# Compare KNN, RandomForest and XGBoost (F1-score) with random search.

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# Some globs

DATA_URL = "./data/data_cleaned.csv"
TARGET = "target"
RANDOM_STATE = 42
N_ITER = 20          # max. number of hyperparameter sets per model
CV_FOLDS = 5
OUTPUT_FILE = "big5.joblib"

# Data

df = pd.read_csv(DATA_URL)
X = df.drop(columns=[TARGET])
y_raw = df[TARGET]

# Encode the target as integers 0..k-1 (I had to learn the hard way XGBoost requires this)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

# F1 averaging: binary if two classes, weighted otherwise
n_classes = len(label_encoder.classes_)
if n_classes == 2:
    f1_average = "binary"
    scoring = "f1"
else:
    f1_average = "weighted"
    scoring = "f1_weighted"

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

# Preprocessor

numeric_cols = X.select_dtypes(include="number").columns.tolist()
categorical_cols = [c for c in X.columns if c not in numeric_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ]
)

# Models and hyperparameter search spaces 


models = {
    "KNeighborsClassifier": (
        KNeighborsClassifier(),
        {
            "model__n_neighbors": randint(3, 31),
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2],
        },
    ),
    "RandomForestClassifier": (
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        {
            "model__n_estimators": randint(100, 501),
            "model__max_depth": [None, 5, 10, 20, 30],
            "model__min_samples_split": randint(2, 11),
            "model__min_samples_leaf": randint(1, 6),
            "model__max_features": ["sqrt", "log2", None],
        },
    ),
       "LogisticRegression": (
        LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
        {
            "model__C": loguniform(1e-3, 1e2),
            "model__solver": ["lbfgs", "saga"],
            "model__class_weight": [None, "balanced"],
        },
    ),
}

# Random search for hyperparmeters

cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
results = {}

for name, (estimator, param_dist) in models.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=N_ITER,
        scoring=scoring,
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)

    y_pred = search.best_estimator_.predict(X_test)
    results[name] = {
        "search": search,
        "cv_f1": search.best_score_,
        "test_f1": f1_score(y_test, y_pred, average=f1_average),
        "params": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
    }

# Output as Markdown to include in handout

print("# Model comparison (F1-score)\n")
print(f"- Rows: {len(df)} (train: {len(X_train)}, test: {len(X_test)})")
print(f"- Classes: {n_classes} -> F1 average: `{f1_average}`")
print(f"- Random search: {N_ITER} parameter sets per model, {CV_FOLDS}-fold CV\n")

print("| Model | CV F1 (train) | Test F1 |")
print("|---|---|---|")
for name, r in results.items():
    print(f"| {name} | {r['cv_f1']:.4f} | {r['test_f1']:.4f} |")
print()

for name, r in results.items():
    print(f"## {name}\n")
    print(f"- CV F1: **{r['cv_f1']:.4f}**")
    print(f"- Test F1: **{r['test_f1']:.4f}**\n")
    print("| Hyperparameter | Value |")
    print("|---|---|")
    for p, v in r["params"].items():
        v = f"{v:.5g}" if isinstance(v, float) else v
        print(f"| {p} | {v} |")
    print()

# Best model = highest cross-validated F1 

best_name = max(results, key=lambda n: results[n]["cv_f1"])
best = results[best_name]
print(f"## Best model: **{best_name}**\n")
print(f"CV F1 = {best['cv_f1']:.4f}, Test F1 = {best['test_f1']:.4f}\n")

# MLflow + joblib export (Here I used Claude)

mlflow.set_experiment("big5")

with mlflow.start_run(run_name=best_name):
    mlflow.log_param("model", best_name)
    mlflow.log_params(best["params"])
    mlflow.log_param("f1_average", f1_average)
    mlflow.log_metric("cv_f1", best["cv_f1"])
    mlflow.log_metric("test_f1", best["test_f1"])
 
    # Save the full pipeline (preprocessor + model) together with the label
    # encoder, so predictions can be mapped back to the original target values.
    joblib.dump(
        {"pipeline": best["search"].best_estimator_, "label_encoder": label_encoder},
        OUTPUT_FILE,
    )
    mlflow.log_artifact(OUTPUT_FILE)
    mlflow.sklearn.log_model(best["search"].best_estimator_, name="model")
 
print(f"Best model saved to `{OUTPUT_FILE}` and logged to MLflow.")
 
