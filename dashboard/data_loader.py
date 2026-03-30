import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from typing import Any
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split


DATA_DIR = Path(__file__).parent.parent / "data"
FINANCE_PATH = DATA_DIR / "finance-report-042024.xlsx"
SENED_PATH = DATA_DIR / "Updated_SENED_Budget_Solidarity_Syria-Turkey_2023_05_17_EN.xlsx"


# Helpers

def _strip_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.astype(str).str.strip()
    return df


def _clean_text(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
        df[col] = df[col].replace("nan", pd.NA)
    return df


def _iqr_outlier_flag(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    flag = ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).astype(int)
    return flag


def _normalize_text(text) -> str:
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _fuzzy_match(a, b, threshold: int = 90) -> int:
    return int(fuzz.token_set_ratio(_normalize_text(a), _normalize_text(b)) >= threshold)


def _tokenize(text) -> list:
    if pd.isna(text):
        return []
    text = re.sub(r"[^a-z0-9]+", " ", str(text).lower())
    return [t for t in text.split() if len(t) >= 2]


def _jaccard_similarity(tokens1: list, tokens2: list) -> float:
    s1, s2 = set(tokens1), set(tokens2)
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def _pairwise_tfidf_cosine(desc1: pd.Series, desc2: pd.Series) -> np.ndarray:
    d1 = desc1.fillna("").astype(str)
    d2 = desc2.fillna("").astype(str)
    vectorizer = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b[a-zA-Z0-9]{2,}\b",
        ngram_range=(1, 2),
        min_df=1,
    )
    vectorizer.fit(pd.concat([d1, d2], axis=0))
    X1 = vectorizer.transform(d1)
    X2 = vectorizer.transform(d2)
    return np.array([cosine_similarity(X1[i], X2[i])[0, 0] for i in range(X1.shape[0])])


# Finance Report

@st.cache_data
def load_finance_report() -> pd.DataFrame:
    df = pd.read_excel(
        FINANCE_PATH,
        sheet_name="Transaction Journal",
        skiprows=2,
        index_col=0,
    )
    df = _strip_columns(df)
    df = df.dropna(how="all")
    df = _clean_text(df)

    for col in ["Transaction Amount", "Total Amount In USD", "Net / Reporting Amount",
                "EX rate", "Charging%", "InforEuro Rate USD to EUR"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


    for col in ["Transaction Date", "Invoice Date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)


    if "Transactions per day" not in df.columns:
        df["Transactions per day"] = df.groupby(
            ["Budget Line Code", "Transaction Date"]
        )["Total Amount In USD"].transform("count")

    if "Average Total Amount In USD" not in df.columns:
        df["Average Total Amount In USD"] = df.groupby(
            ["Budget Line Code", "Transaction Date"]
        )["Total Amount In USD"].transform("mean")

    if "Relative deviation" not in df.columns:
        avg = df["Average Total Amount In USD"]
        df["Relative deviation"] = (df["Total Amount In USD"] - avg) / avg.replace(0, np.nan)

    if "Velocity Score" not in df.columns:
        df["Velocity Score"] = (
            np.log1p(df["Transactions per day"]) + df["Relative deviation"]
        )


    # Combined description similarity: RapidFuzz 60% + TF-IDF 30% + Jaccard 10%
    recip_col = next(
        (c for c in df.columns if "2nd Description" in c or "Recipent" in c or "Recipient" in c),
        None,
    )
    if recip_col and "description_match_final" not in df.columns:
        rf_scores = df.apply(
            lambda r: fuzz.token_set_ratio(
                _normalize_text(r.get("Description", "")),
                _normalize_text(r.get(recip_col, "")),
            ),
            axis=1,
        ).astype(float)
        tfidf_cos = _pairwise_tfidf_cosine(df["Description"], df[recip_col])
        tokens1 = df["Description"].apply(_tokenize)
        tokens2 = df[recip_col].apply(_tokenize)
        jaccard = pd.Series(
            [_jaccard_similarity(a, b) for a, b in zip(tokens1, tokens2)],
            index=df.index,
        )
        df["rapidfuzz_score"] = rf_scores
        df["tfidf_cosine"] = tfidf_cos
        df["jaccard_score"] = jaccard
        df["similarity_combined"] = (
            0.6 * (rf_scores / 100) + 0.3 * tfidf_cos + 0.1 * jaccard
        )
        df["description_match_final"] = (
            (rf_scores >= 90) | (tfidf_cos >= 0.25) | (df["similarity_combined"] >= 0.65)
        ).astype(int)
        df["description_fraude_signal"] = (df["description_match_final"] == 0).astype(int)
        # keep match_description as alias for backward compat
        df["match_description"] = df["description_match_final"]

    df["is_outlier"] = _iqr_outlier_flag(df["Total Amount In USD"])

    df["invoice_missing"] = df["Invoice Date"].isna().astype(int)

    return df


# SENED Transactions

_SENED_RENAME_FRAGMENTS = {
    "Budget": "budget",
    "REF": "ref",
    "Date of": "date_voucher",
    "Name of the Recipient": "recipient",
    "Reason for payment": "reason",
    "Currency": "currency",
    "Amount": "amount",
    "Expenditure": "amount_eur",
    "Year": "year",
    "Period": "period",
}


def _rename_sened_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        col_clean = col.replace("\n", " ").strip()
        for fragment, alias in _SENED_RENAME_FRAGMENTS.items():
            if col_clean.startswith(fragment) and alias not in rename_map.values():
                rename_map[col] = alias
                break
    return df.rename(columns=rename_map)


@st.cache_data
def load_sened_transactions() -> pd.DataFrame:
    df = pd.read_excel(
        SENED_PATH,
        sheet_name="Transactions List",
        skiprows=3,
        index_col=0,
    )
    df = df.dropna(how="all")
    df = _rename_sened_columns(df)
    df = _clean_text(df)

    if "budget" in df.columns:
        df["budget"] = (
            df["budget"]
            .astype(str)
            .str.strip()
            .str.replace(r"\.0$", "", regex=True)
            .replace("nan", pd.NA)
        )

    for col in ["amount", "amount_eur", "period", "year"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "date_voucher" in df.columns:
        df["date_voucher"] = pd.to_datetime(df["date_voucher"], errors="coerce", dayfirst=True)


    df = df.dropna(subset=["amount"], how="all")


    df["is_outlier"] = _iqr_outlier_flag(df["amount"])

    if "ref" in df.columns:
        ref_counts = df["ref"].value_counts()
        df["ref_duplicate"] = df["ref"].map(ref_counts > 1).fillna(False).astype(int)
    else:
        df["ref_duplicate"] = 0

    return df


# Flag filters

def get_finance_flags(df: pd.DataFrame) -> pd.DataFrame:
    vel_threshold = df["Velocity Score"].quantile(0.75)

    fraud_col = "description_fraude_signal" if "description_fraude_signal" in df.columns else "match_description"
    conditions = {
        "high_velocity": df["Velocity Score"] > vel_threshold,
        "description_fraude": df[fraud_col] == 1 if fraud_col == "description_fraude_signal" else df[fraud_col] == 0,
        "outlier": df["is_outlier"] == 1,
        "invoice_missing": df["invoice_missing"] == 1,
    }

    mask = conditions["high_velocity"] | conditions["description_fraude"] | conditions["outlier"] | conditions["invoice_missing"]
    flagged = df[mask].copy()

    flagged["flags"] = flagged.apply(
        lambda r: ", ".join(
            k for k, cond in conditions.items() if cond.loc[r.name]
        ),
        axis=1,
    )
    return flagged


def get_sened_flags(df: pd.DataFrame) -> pd.DataFrame:
    conditions = {
        "outlier": df["is_outlier"] == 1,
        "ref_duplicate": df["ref_duplicate"] == 1,
    }

    mask = conditions["outlier"] | conditions["ref_duplicate"]
    flagged = df[mask].copy()

    flagged["flags"] = flagged.apply(
        lambda r: ", ".join(
            k for k, cond in conditions.items() if cond.loc[r.name]
        ),
        axis=1,
    )
    return flagged

# ─── Linear Regression Model ────────────────────────────────────────────────
 
REGRESSION_FEATURES = ["Total Amount In USD", "Transactions per day", "Relative deviation"]
REGRESSION_TARGET   = "Velocity Score"
 
 
@st.cache_data
def get_regression_model(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train a LinearRegression model to predict Velocity Score.
 
    Returns a dict with:
        model         – fitted LinearRegression
        X_test        – test features (DataFrame)
        y_test        – true target values (Series)
        y_pred_test   – predicted values on test set (ndarray)
        y_pred_train  – predicted values on train set (ndarray)
        r2_train      – R² on training set
        r2_test       – R² on test set
        rmse_test     – RMSE on test set
        feature_names – list of feature column names
        coefficients  – dict {feature: coef}
        intercept     – model intercept
    """
    required = REGRESSION_FEATURES + [REGRESSION_TARGET]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for regression: {missing}")
 
    clean = df[required].dropna()
    if len(clean) < 10:
        raise ValueError("Not enough data after dropping NaNs (need at least 10 rows).")
 
    X = clean[REGRESSION_FEATURES]
    y = clean[REGRESSION_TARGET]
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
 
    model = LinearRegression()
    model.fit(X_train, y_train)
 
    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)
 
    return {
        "model":         model,
        "X_test":        X_test,
        "y_test":        y_test,
        "y_pred_test":   y_pred_test,
        "y_pred_train":  y_pred_train,
        "r2_train":      r2_score(y_train, y_pred_train),
        "r2_test":       r2_score(y_test, y_pred_test),
        "rmse_test":     root_mean_squared_error(y_test, y_pred_test),
        "feature_names": REGRESSION_FEATURES,
        "coefficients":  dict(zip(REGRESSION_FEATURES, model.coef_)),
        "intercept":     model.intercept_,
    }