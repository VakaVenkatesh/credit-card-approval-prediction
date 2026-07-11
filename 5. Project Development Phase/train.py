import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

DATA_DIR = "data"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Step 1: Load
# ---------------------------------------------------------------------------
application_df = pd.read_csv(f"{DATA_DIR}/application_record.csv")
credit_df = pd.read_csv(f"{DATA_DIR}/credit_record.csv")

print(f"application_record: {application_df.shape}, credit_record: {credit_df.shape}")

# ---------------------------------------------------------------------------
# Step 2: Clean
# ---------------------------------------------------------------------------
application_df = application_df.drop_duplicates(subset="ID", keep="first").reset_index(drop=True)
application_df["OCCUPATION_TYPE"] = application_df["OCCUPATION_TYPE"].fillna("Not Employed")

# ---------------------------------------------------------------------------
# Step 3: Target - ever 60+ days overdue (STATUS 2-5) = high risk
# ---------------------------------------------------------------------------
high_risk_statuses = ["2", "3", "4", "5"]
credit_df["IS_HIGH_RISK"] = credit_df["STATUS"].isin(high_risk_statuses).astype(int)
risk_summary = credit_df.groupby("ID")["IS_HIGH_RISK"].max().reset_index()
risk_summary.rename(columns={"IS_HIGH_RISK": "TARGET"}, inplace=True)

merged_df = pd.merge(application_df, risk_summary, on="ID", how="inner")
merged_df.drop(columns=["FLAG_MOBIL"], inplace=True, errors="ignore")

print(f"Merged shape: {merged_df.shape}, positive rate: {merged_df['TARGET'].mean():.4f}")

# ---------------------------------------------------------------------------
# Step 4: Feature engineering (same as original + 2 new affordability signals)
# ---------------------------------------------------------------------------
merged_df["AGE_YEARS"] = (-merged_df["DAYS_BIRTH"] / 365).astype(int)
merged_df["IS_EMPLOYED"] = (merged_df["DAYS_EMPLOYED"] != 365243).astype(int)
merged_df["YEARS_EMPLOYED"] = np.where(
    merged_df["DAYS_EMPLOYED"] != 365243, -merged_df["DAYS_EMPLOYED"] / 365, 0
)
merged_df.drop(columns=["DAYS_BIRTH", "DAYS_EMPLOYED"], inplace=True)

merged_df["INCOME_PER_PERSON"] = merged_df["AMT_INCOME_TOTAL"] / merged_df["CNT_FAM_MEMBERS"]
merged_df["LOG_INCOME"] = np.log1p(merged_df["AMT_INCOME_TOTAL"])

# NEW: employment stability ratio - how much of the applicant's life has been
# spent employed. Zero for pensioners/unemployed, low for new hires, higher
# for long-tenured workers. Weakly available before but never surfaced as its
# own feature.
merged_df["EMPLOYMENT_STABILITY"] = np.where(
    merged_df["AGE_YEARS"] > 0, merged_df["YEARS_EMPLOYED"] / merged_df["AGE_YEARS"], 0
)

occ_counts = merged_df["OCCUPATION_TYPE"].value_counts(normalize=True)
rare_occupations = occ_counts[occ_counts < 0.01].index
merged_df["OCCUPATION_TYPE"] = merged_df["OCCUPATION_TYPE"].apply(
    lambda x: "Other" if x in rare_occupations else x
)
occupation_model_categories = sorted(merged_df["OCCUPATION_TYPE"].unique().tolist())

merged_df["CONTACT_SCORE"] = (
    merged_df["FLAG_WORK_PHONE"] + merged_df["FLAG_PHONE"] + merged_df["FLAG_EMAIL"]
)

# ---------------------------------------------------------------------------
# Step 5: Encoding
# ---------------------------------------------------------------------------
binary_cols = ["CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY"]
multi_cat_cols = [
    "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "OCCUPATION_TYPE",
]

label_encoders = {}
for col in binary_cols:
    le = LabelEncoder()
    merged_df[col] = le.fit_transform(merged_df[col])
    label_encoders[col] = dict(zip(le.classes_, le.transform(le.classes_)))

merged_df = pd.get_dummies(merged_df, columns=multi_cat_cols, drop_first=True)

X = merged_df.drop(columns=["ID", "TARGET"])
y = merged_df["TARGET"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
scale_pos_weight = neg / pos
print(f"scale_pos_weight: {scale_pos_weight:.2f}")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# ---------------------------------------------------------------------------
# Step 6: Baseline model comparison (kept for the README table / narrative)
# ---------------------------------------------------------------------------
results = []

lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
lr.fit(X_train, y_train)
results.append(("Logistic Regression", lr, lr.predict_proba(X_test)[:, 1]))

dt = DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=RANDOM_STATE)
dt.fit(X_train, y_train)
results.append(("Decision Tree", dt, dt.predict_proba(X_test)[:, 1]))

rf = RandomForestClassifier(
    n_estimators=300, max_depth=10, class_weight="balanced",
    random_state=RANDOM_STATE, n_jobs=-1,
)
rf.fit(X_train, y_train)
results.append(("Random Forest", rf, rf.predict_proba(X_test)[:, 1]))

xgb_base = XGBClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    scale_pos_weight=scale_pos_weight, eval_metric="logloss",
    random_state=RANDOM_STATE, n_jobs=-1,
)
xgb_base.fit(X_train, y_train)
results.append(("XGBoost (untuned)", xgb_base, xgb_base.predict_proba(X_test)[:, 1]))

comparison_rows = []
for name, model, proba in results:
    pred = (proba >= 0.5).astype(int)
    comparison_rows.append({
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, pred), 3),
        "ROC_AUC": round(roc_auc_score(y_test, proba), 3),
        "F1_high_risk": round(f1_score(y_test, pred), 3),
    })

# ---------------------------------------------------------------------------
# Step 7: Hyperparameter tuning (XGBoost)
# ---------------------------------------------------------------------------
param_dist = {
    "n_estimators": [200, 300, 400, 600],
    "max_depth": [3, 4, 5, 6, 8],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
    "min_child_weight": [1, 3, 5],
    "scale_pos_weight": [scale_pos_weight * f for f in [0.5, 0.75, 1.0, 1.25, 1.5]],
}

search = RandomizedSearchCV(
    estimator=XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1),
    param_distributions=param_dist,
    n_iter=15,
    scoring="roc_auc",
    cv=skf,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=1,
)
search.fit(X_train, y_train)
print("Best CV ROC-AUC:", search.best_score_)
print("Best params:", search.best_params_)

final_model = search.best_estimator_
final_model.fit(X_train, y_train)

y_proba_final = final_model.predict_proba(X_test)[:, 1]
y_pred_default = (y_proba_final >= 0.5).astype(int)

comparison_rows.append({
    "Model": "XGBoost (tuned, threshold=0.5)",
    "Accuracy": round(accuracy_score(y_test, y_pred_default), 3),
    "ROC_AUC": round(roc_auc_score(y_test, y_proba_final), 3),
    "F1_high_risk": round(f1_score(y_test, y_pred_default), 3),
})

# ---------------------------------------------------------------------------
# Step 8: Threshold tuning - maximize F2 (favors recall) instead of using 0.5
#
# Important: the threshold is chosen on a VALIDATION split carved out of the
# training data (not the final test set, and not by refitting on all of
# X_train first). final_model is refit on train_sub only, scored on val_sub
# to pick the threshold, and then refit on the FULL X_train afterwards for
# the artifact that actually gets deployed. This keeps the test set fully
# held out for a single, final, unbiased evaluation.
# ---------------------------------------------------------------------------
from sklearn.metrics import precision_score, recall_score

X_train_sub, X_val, y_train_sub, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=RANDOM_STATE, stratify=y_train
)

threshold_model = XGBClassifier(**search.best_params_, eval_metric="logloss",
                                 random_state=RANDOM_STATE, n_jobs=-1)
threshold_model.fit(X_train_sub, y_train_sub)
val_proba = threshold_model.predict_proba(X_val)[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y_val, val_proba)

# Policy: maximize F2 (recall weighted 2x precision) on the validation split.
# We tried a hard recall floor (e.g. >=0.65 recall) first, but that pushed the
# threshold down to ~0.05 and flagged >50% of ALL applicants for review just
# to catch that many positives - at that point the "triage" tool isn't saving
# analysts any work over reviewing everyone. F2 lands at a much more useful
# operating point: flag a small slice of applicants, concentrated with risk.
f2_scores = np.zeros_like(thresholds)
for i, t in enumerate(thresholds):
    pred_t = (val_proba >= t).astype(int)
    f2_scores[i] = fbeta_score(y_val, pred_t, beta=2, zero_division=0)

best_idx = int(np.argmax(f2_scores))
best_threshold = float(thresholds[best_idx])
flag_rate = float((val_proba >= best_threshold).mean())
print(f"Validation precision/recall/F2 at chosen threshold: "
      f"{precisions[best_idx]:.3f} / {recalls[best_idx]:.3f} / {f2_scores[best_idx]:.3f}")
print(f"Share of validation applicants flagged at this threshold: {flag_rate:.1%}")
print(f"Chosen decision threshold (max F2, validation split): {best_threshold:.4f}")

print("  threshold | precision | recall")
for t in sorted(set([0.1, 0.2, 0.3, 0.4, 0.5, round(best_threshold, 3)])):
    p = (val_proba >= t).astype(int)
    print(f"  {t:.3f}     | {precision_score(y_val, p, zero_division=0):.3f}     | {recall_score(y_val, p, zero_division=0):.3f}")

y_pred_tuned_threshold = (y_proba_final >= best_threshold).astype(int)
comparison_rows.append({
    "Model": f"XGBoost (tuned, threshold={best_threshold:.2f})",
    "Accuracy": round(accuracy_score(y_test, y_pred_tuned_threshold), 3),
    "ROC_AUC": round(roc_auc_score(y_test, y_proba_final), 3),
    "F1_high_risk": round(f1_score(y_test, y_pred_tuned_threshold), 3),
})

print("\n=== Model comparison ===")
comparison_df = pd.DataFrame(comparison_rows)
print(comparison_df.to_string(index=False))

print("\n=== Final classification report (tuned threshold) ===")
report = classification_report(y_test, y_pred_tuned_threshold, output_dict=True)
print(classification_report(y_test, y_pred_tuned_threshold))

# 5-fold CV on the final config, for an honest generalization estimate
cv_scores = cross_val_score(final_model, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)
print(f"\n5-fold CV ROC-AUC: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

# ---------------------------------------------------------------------------
# Step 9: Global feature importance (for README + UI "what drives risk" panel)
# ---------------------------------------------------------------------------
importances = pd.Series(final_model.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(12)
print("\n=== Top 12 features by importance ===")
print(top_features)

# ---------------------------------------------------------------------------
# Step 10: Save all artifacts
# ---------------------------------------------------------------------------
joblib.dump(final_model, f"{MODEL_DIR}/best_model.pkl")
joblib.dump(X.columns.tolist(), f"{MODEL_DIR}/model_columns.pkl")
joblib.dump(best_threshold, f"{MODEL_DIR}/decision_threshold.pkl")

categorical_value_map = {
    "NAME_INCOME_TYPE": application_df["NAME_INCOME_TYPE"].unique().tolist(),
    "NAME_EDUCATION_TYPE": application_df["NAME_EDUCATION_TYPE"].unique().tolist(),
    "NAME_FAMILY_STATUS": application_df["NAME_FAMILY_STATUS"].unique().tolist(),
    "NAME_HOUSING_TYPE": application_df["NAME_HOUSING_TYPE"].unique().tolist(),
    "OCCUPATION_TYPE": sorted(application_df["OCCUPATION_TYPE"].fillna("Not Employed").unique().tolist()),
    "OCCUPATION_TYPE_MODEL_CATEGORIES": occupation_model_categories,
}
joblib.dump(categorical_value_map, f"{MODEL_DIR}/categorical_value_map.pkl")

with open(f"{MODEL_DIR}/metrics.json", "w") as f:
    json.dump({
        "comparison_table": comparison_rows,
        "classification_report": report,
        "cv_roc_auc_mean": float(cv_scores.mean()),
        "cv_roc_auc_std": float(cv_scores.std()),
        "decision_threshold": best_threshold,
        "validation_flag_rate": flag_rate,
        "top_features": {k: float(v) for k, v in top_features.items()},
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "positive_rate": float(y.mean()),
    }, f, indent=2)

print("\nSaved: best_model.pkl, model_columns.pkl, categorical_value_map.pkl, decision_threshold.pkl, metrics.json")
