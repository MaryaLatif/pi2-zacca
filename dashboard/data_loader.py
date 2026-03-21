import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from rapidfuzz import fuzz

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


    if "match_description" not in df.columns:
        df["match_description"] = df.apply(
            lambda r: _fuzzy_match(
                r.get("Description", ""),
                r.get("2nd Description (Recipient)", ""),
            ),
            axis=1,
        )

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

    conditions = {
        "high_velocity": df["Velocity Score"] > vel_threshold,
        "mismatch": df["match_description"] == 0,
        "outlier": df["is_outlier"] == 1,
        "invoice_missing": df["invoice_missing"] == 1,
    }

    mask = conditions["high_velocity"] | conditions["mismatch"] | conditions["outlier"] | conditions["invoice_missing"]
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
