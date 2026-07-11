# app/app.py

from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import json
import os

import rules

app = Flask(__name__)

# --- Load model artifacts (saved from train.py) ---
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")

best_model = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
model_columns = joblib.load(os.path.join(MODEL_DIR, "model_columns.pkl"))
categorical_value_map = joblib.load(os.path.join(MODEL_DIR, "categorical_value_map.pkl"))
DECISION_THRESHOLD = joblib.load(os.path.join(MODEL_DIR, "decision_threshold.pkl"))

with open(os.path.join(MODEL_DIR, "metrics.json")) as f:
    METRICS = json.load(f)

# SHAP explainer, built once at startup - reused across requests
try:
    import shap
    EXPLAINER = shap.TreeExplainer(best_model)
except Exception:
    EXPLAINER = None

# Friendly labels for the raw column names SHAP will point to
FEATURE_LABELS = {
    "AMT_INCOME_TOTAL": "Annual income",
    "LOG_INCOME": "Annual income (log-scaled)",
    "INCOME_PER_PERSON": "Income per family member",
    "AGE_YEARS": "Age",
    "YEARS_EMPLOYED": "Years employed",
    "IS_EMPLOYED": "Employment status",
    "EMPLOYMENT_STABILITY": "Employment stability (years employed / age)",
    "CNT_CHILDREN": "Number of children",
    "CNT_FAM_MEMBERS": "Family size",
    "CONTACT_SCORE": "Contact details on file",
    "CODE_GENDER": "Gender",
    "FLAG_OWN_CAR": "Car ownership",
    "FLAG_OWN_REALTY": "Property ownership",
    "FLAG_WORK_PHONE": "Work phone on file",
    "FLAG_PHONE": "Phone on file",
    "FLAG_EMAIL": "Email on file",
}


def friendly_feature_name(col: str) -> str:
    if col in FEATURE_LABELS:
        return FEATURE_LABELS[col]
    for prefix, label in [
        ("NAME_INCOME_TYPE_", "Income type"),
        ("NAME_EDUCATION_TYPE_", "Education level"),
        ("NAME_FAMILY_STATUS_", "Family status"),
        ("NAME_HOUSING_TYPE_", "Housing type"),
        ("OCCUPATION_TYPE_", "Occupation"),
    ]:
        if col.startswith(prefix):
            value = col[len(prefix):].replace("_", " ")
            return f"{label}: {value}"
    return col


def risk_band(probability: float) -> str:
    """Three-tier band relative to the tuned decision threshold, not a flat 50%."""
    if probability >= DECISION_THRESHOLD:
        return "high"
    if probability >= DECISION_THRESHOLD * 0.5:
        return "medium"
    return "low"


@app.route("/")
def home():
    return render_template("home.html", metrics=METRICS)


@app.route("/predict", methods=["GET"])
def predict_form():
    return render_template(
        "predict.html",
        income_types=categorical_value_map["NAME_INCOME_TYPE"],
        education_types=categorical_value_map["NAME_EDUCATION_TYPE"],
        family_statuses=categorical_value_map["NAME_FAMILY_STATUS"],
        housing_types=categorical_value_map["NAME_HOUSING_TYPE"],
        occupation_types=categorical_value_map["OCCUPATION_TYPE"],
    )


@app.route("/predict", methods=["POST"])
def predict():
    form = request.form
    errors = []

    def parse_float(name, label):
        try:
            return float(form[name])
        except (ValueError, KeyError):
            errors.append(f"{label} must be a number.")
            return 0.0

    def parse_int(name, label):
        try:
            return int(form[name])
        except (ValueError, KeyError):
            errors.append(f"{label} must be a whole number.")
            return 0

    income = parse_float("income", "Annual income")
    age = parse_int("age", "Age")
    years_employed = parse_float("years_employed", "Years employed")
    family_members = parse_float("family_members", "Family members")
    children = parse_int("children", "Number of children")

    if family_members < 1:
        errors.append("Family members must be at least 1.")
        family_members = 1

    if errors:
        return render_template("result.html", form_errors=errors)

    raw_input = {
        "CODE_GENDER": form["gender"],
        "FLAG_OWN_CAR": form["own_car"],
        "FLAG_OWN_REALTY": form["own_realty"],
        "CNT_CHILDREN": children,
        "AMT_INCOME_TOTAL": income,
        "NAME_INCOME_TYPE": form["income_type"],
        "NAME_EDUCATION_TYPE": form["education"],
        "NAME_FAMILY_STATUS": form["family_status"],
        "NAME_HOUSING_TYPE": form["housing_type"],
        "OCCUPATION_TYPE": form["occupation"],
        "CNT_FAM_MEMBERS": family_members,
        "FLAG_WORK_PHONE": 1 if form.get("work_phone") else 0,
        "FLAG_PHONE": 1 if form.get("phone") else 0,
        "FLAG_EMAIL": 1 if form.get("email") else 0,
        "AGE_YEARS": age,
        "YEARS_EMPLOYED": years_employed,
    }
    raw_input["IS_EMPLOYED"] = 1 if raw_input["YEARS_EMPLOYED"] > 0 else 0

    # --- STEP 1: hard rule gate, runs BEFORE any ML call ---
    rule_result = rules.evaluate(raw_input)

    if rule_result.hard_decline:
        return render_template(
            "result.html",
            decision="decline",
            band="high",
            source="rules",
            reasons=rule_result.reasons,
            probability=None,
            threshold=DECISION_THRESHOLD,
        )

    # --- STEP 2: ML risk scoring (only for applicants that passed the gate) ---
    occupation_categories = categorical_value_map.get(
        "OCCUPATION_TYPE_MODEL_CATEGORIES", categorical_value_map["OCCUPATION_TYPE"]
    )
    if raw_input["OCCUPATION_TYPE"] not in occupation_categories:
        raw_input["OCCUPATION_TYPE"] = "Other"

    raw_input["INCOME_PER_PERSON"] = raw_input["AMT_INCOME_TOTAL"] / raw_input["CNT_FAM_MEMBERS"]
    raw_input["LOG_INCOME"] = np.log1p(raw_input["AMT_INCOME_TOTAL"])
    raw_input["CONTACT_SCORE"] = (
        raw_input["FLAG_WORK_PHONE"] + raw_input["FLAG_PHONE"] + raw_input["FLAG_EMAIL"]
    )
    raw_input["EMPLOYMENT_STABILITY"] = (
        raw_input["YEARS_EMPLOYED"] / raw_input["AGE_YEARS"] if raw_input["AGE_YEARS"] > 0 else 0
    )

    raw_input["CODE_GENDER"] = 1 if raw_input["CODE_GENDER"] == "M" else 0
    raw_input["FLAG_OWN_CAR"] = 1 if raw_input["FLAG_OWN_CAR"] == "Y" else 0
    raw_input["FLAG_OWN_REALTY"] = 1 if raw_input["FLAG_OWN_REALTY"] == "Y" else 0

    input_df = pd.DataFrame([raw_input])
    multi_cat_cols = ["NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
                      "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "OCCUPATION_TYPE"]
    input_df = pd.get_dummies(input_df, columns=multi_cat_cols)
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    probability = float(best_model.predict_proba(input_df)[0][1])
    band = risk_band(probability)
    decision = "review" if probability >= DECISION_THRESHOLD else "clear"

    # --- STEP 3: explain the score (top SHAP contributors pushing risk up) ---
    top_reasons = []
    if EXPLAINER is not None:
        try:
            shap_values = EXPLAINER.shap_values(input_df)
            row = shap_values[0] if shap_values.ndim == 2 else shap_values
            input_row = input_df.iloc[0]

            one_hot_prefixes = (
                "NAME_INCOME_TYPE_", "NAME_EDUCATION_TYPE_",
                "NAME_FAMILY_STATUS_", "NAME_HOUSING_TYPE_", "OCCUPATION_TYPE_",
            )

            def is_relevant(col):
                # A one-hot column only describes the applicant if it's actually
                # set to 1 for them - a 0 there just means "not this category",
                # which isn't a meaningful reason to show as-is.
                if col.startswith(one_hot_prefixes):
                    return input_row[col] == 1
                return True

            contributions = [
                (col, val) for col, val in zip(input_df.columns, row) if is_relevant(col)
            ]
            contributions.sort(key=lambda x: x[1], reverse=True)
            for col, val in contributions[:4]:
                if val > 0:
                    top_reasons.append(f"{friendly_feature_name(col)} increases estimated risk.")
            for col, val in sorted(contributions, key=lambda x: x[1])[:2]:
                if val < 0:
                    top_reasons.append(f"{friendly_feature_name(col)} lowers estimated risk.")
        except Exception:
            top_reasons = []

    return render_template(
        "result.html",
        decision=decision,
        band=band,
        source="model",
        reasons=top_reasons,
        soft_flags=rule_result.soft_flags,
        probability=round(probability * 100, 1),
        threshold=round(DECISION_THRESHOLD * 100, 1),
    )


if __name__ == "__main__":
    app.run(debug=True)
