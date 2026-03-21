import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from dashboard.data_loader import (
    get_finance_flags,
    get_sened_flags,
    load_finance_report,
    load_sened_transactions,
)

# Config

st.set_page_config(
    page_title="Tableau de bord Zacca",
    page_icon="🔎",
    layout="wide",
)

OUTLIER_COLORS = {0: "#3a86ff", 1: "#ff006e"}   # bleu = normal, rouge = anomalie
MATCH_COLORS   = {0: "#ff006e", 1: "#3a86ff"}   # rouge = discordance, bleu = concordance

# Chargement des données


with st.spinner("Chargement des données..."):
    fr = load_finance_report()
    sened = load_sened_transactions()



st.title("Tableau de bord Zacca — Audit Financier")
st.caption("Signaux d'anomalie sur les transactions financières — Rapport Financier Avril 2024 & SENED Syrie-Turquie")

# Barre latérale — Documentation des données

with st.sidebar:
    st.header("À propos des données")

    st.subheader("Jeux de données")
    st.markdown("""
**Rapport Financier** (`finance-report-042024.xlsx`)
Transactions d'avril 2024 — ~161 lignes, 23 colonnes.
Données sources : montants, dates, catégories, statut de paiement, descriptions, bénéficiaires.

**Transactions SENED** (`Updated_SENED_...xlsx`)
Transactions du projet Solidarité Syrie-Turquie — ~270 lignes.
Données sources : références comptables, bénéficiaires, raisons de paiement, montants par devise et période.
""")

    st.subheader("Colonnes calculées au chargement")
    st.markdown("""
Ces colonnes **ne figurent pas dans les fichiers Excel** — elles sont recalculées à chaque chargement.

**Répliquées depuis `project.ipynb`** :

| Colonne | Calcul |
|---|---|
| `Velocity Score` | `ln(nb tx/jour + 1) + écart relatif` |
| `Transactions per day` | Nb de tx partageant la même ligne budgétaire + date |
| `Relative deviation` | `(montant − moyenne) / moyenne` par ligne budgétaire + date |
| `match_description` | Score RapidFuzz ≥ 90 entre `Description` et `Bénéficiaire` → 0/1 |
| `is_outlier` | IQR sur le montant : hors `[Q1−1,5×IQR ; Q3+1,5×IQR]` → 0/1 |

**Nouvelles colonnes ajoutées pour le dashboard** :

| Colonne | Calcul |
|---|---|
| `invoice_missing` | 1 si `Invoice Date` est absent, 0 sinon |
| `ref_duplicate` | 1 si la REF apparaît plus d'une fois dans le SENED |
""")

    st.subheader("Structure des codes budget SENED")
    st.markdown("""
Chaque transaction SENED est rattachée à une ligne budgétaire prédéfinie du projet :

| Code | Catégorie |
|---|---|
| `1.1.xx` | Ressources humaines (salaires par rôle) |
| `2.x` | Frais de déplacement (véhicules, carburant, voyages) |
| `3.1.x` | Matériels et fournitures (kits, outils, impression) |
| `4.2` | Frais de bureau courants |
| `4.3` | Consultants externes |
| `4.5` | Gestion de cas *(le plus concentré dans les données)* |

Une concentration anormale sur un code (ex. `4.5`) ou une devise inattendue (USD sur des charges locales) signale une potentielle mauvaise imputation ou un détournement.
""")

    st.subheader("Convention couleur")
    st.markdown("""
- 🔵 **Bleu** → transaction normale
- 🔴 **Rouge** → anomalie / signal de fraude
- 🟠 **Ligne orange pointillée** → seuil ajustable (Velocity Score)
""")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Vue d'ensemble", "Rapport Financier", "Transactions SENED", "Transactions Suspectes"]
)

# ONGLET 1 — Vue d'ensemble

with tab1:
    st.info(
        "**Vue d'ensemble** — Résumé des deux jeux de données. "
        "Les 5 indicateurs en haut synthétisent les principaux signaux de risque détectés automatiquement. "
        "Les graphiques donnent une lecture rapide de la répartition temporelle et budgétaire des dépenses."
    )

    # KPIs
    total_fr = len(fr)
    total_usd = fr["Total Amount In USD"].sum()
    pct_outliers = (
        (fr["is_outlier"].sum() + sened["is_outlier"].sum())
        / (len(fr) + len(sened))
        * 100
    )
    pct_mismatch = fr["match_description"].eq(0).sum() / total_fr * 100
    dup_ref = int(sened["ref_duplicate"].sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Transactions (Rapport Financier)", total_fr)
    c2.metric("Total USD (Rapport Financier)", f"${total_usd:,.0f}")
    c3.metric("% Valeurs aberrantes (deux jeux)", f"{pct_outliers:.1f}%")
    c4.metric("% Discordances description", f"{pct_mismatch:.1f}%")
    c5.metric("REF en double (SENED)", dup_ref)

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        daily = (
            fr.groupby(["Transaction Date", "Category (optional)"])["Total Amount In USD"]
            .sum()
            .reset_index()
        )
        daily.columns = ["Date", "Catégorie", "Total USD"]
        fig = px.bar(
            daily,
            x="Date",
            y="Total USD",
            color="Catégorie",
            title="Volume journalier des transactions par catégorie (Rapport Financier)",
            labels={"Total USD": "Montant total (USD)"},
        )
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "📂 Rapport Financier · Variables : `Transaction Date`, `Total Amount In USD`, `Category` · "
            "Un pic sur une seule date indique une concentration anormale de dépenses à surveiller."
        )

    with col_right:
        payment_counts = fr["Payment Status"].value_counts().reset_index()
        payment_counts.columns = ["Statut", "Nombre"]
        fig = px.pie(
            payment_counts,
            names="Statut",
            values="Nombre",
            title="Répartition du statut de paiement (Rapport Financier)",
            color_discrete_sequence=["#3a86ff", "#ff006e"],
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "📂 Rapport Financier · Variable : `Payment Status` · "
            "Une part d'accru élevée (>60 %) signifie des dépenses comptabilisées mais non encore réglées."
        )

    if "period" in sened.columns and "amount_eur" in sened.columns:
        period_df = sened.copy()
        period_df["period_label"] = period_df.apply(
            lambda r: f"P{int(r['period'])}" if pd.notna(r["period"]) else "?", axis=1
        )
        period_grp = (
            period_df.groupby(["period_label", "budget"])["amount_eur"]
            .sum()
            .reset_index()
        )
        fig = px.bar(
            period_grp.sort_values("period_label"),
            x="period_label",
            y="amount_eur",
            color="budget",
            title="Dépenses SENED par période et ligne budgétaire",
            labels={"period_label": "Période", "amount_eur": "Montant (EUR)"},
        )
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "📂 SENED · Variables : `period`, `amount_eur`, `budget` · "
            "Des pics concentrés sur une ou deux périodes (notamment P1, P2, P11) indiquent des décaissements précipités."
        )

# ONGLET 2 — Rapport Financier

with tab2:
    st.subheader("Rapport Financier — Signaux de fraude")
    st.info(
        "**Source :** `finance-report-042024.xlsx` — Transactions d'avril 2024 (~161 lignes). "
        "Quatre signaux ont été calculés sur ce jeu : "
        "le **Velocity Score** (concentration des tx par ligne budgétaire et par jour), "
        "la **discordance de description** (comparaison textuelle Description ↔ Bénéficiaire via RapidFuzz), "
        "les **valeurs aberrantes** (méthode IQR sur le montant en USD) "
        "et les **factures manquantes** (absence de `Invoice Date`). "
        "Aucune étiquette de fraude confirmée n'existe — ces signaux servent à **prioriser les lignes à examiner**."
    )

    f_col1, f_col2 = st.columns([2, 1])
    with f_col1:
        budget_options = sorted(fr["Budget Line Code"].dropna().unique().tolist())
        selected_budgets = st.multiselect(
            "Filtrer par ligne budgétaire",
            options=budget_options,
            default=budget_options,
            key="fr_budget_filter",
        )
    with f_col2:
        vel_q75 = float(fr["Velocity Score"].quantile(0.75))
        vel_threshold = st.slider(
            "Seuil Velocity Score (risque élevé)",
            min_value=float(fr["Velocity Score"].min()),
            max_value=float(fr["Velocity Score"].max()),
            value=vel_q75,
            step=0.05,
            key="vel_threshold",
        )

    fr_filtered = fr[fr["Budget Line Code"].isin(selected_budgets)].copy()
    fr_filtered["Risque"] = fr_filtered["Velocity Score"].apply(
        lambda v: "Vélocité élevée" if v > vel_threshold else "Normal"
    )
    fr_filtered["is_outlier_label"] = fr_filtered["is_outlier"].map({0: "Normal", 1: "Aberrant"})
    fr_filtered["match_label"] = fr_filtered["match_description"].map(
        {0: "Discordance", 1: "Concordance"}
    )

    c1, c2 = st.columns(2)

    with c1:
        fig = px.scatter(
            fr_filtered,
            x="Total Amount In USD",
            y="Velocity Score",
            color="is_outlier_label",
            color_discrete_map={"Normal": "#3a86ff", "Aberrant": "#ff006e"},
            hover_data=["Description", "Budget Line Code", "Transaction Date"],
            title="Velocity Score vs Montant — Valeurs aberrantes",
            labels={"is_outlier_label": ""},
        )
        fig.add_hline(
            y=vel_threshold,
            line_dash="dash",
            line_color="orange",
            annotation_text=f"Seuil ({vel_threshold:.2f})",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "📂 Rapport Financier · Variables : `Total Amount In USD`, `Velocity Score`, `is_outlier` · "
            "Les points rouges au-dessus du seuil (montant élevé + vélocité élevée) sont les plus prioritaires pour l'audit."
        )

    with c2:
        mismatch_grp = (
            fr_filtered.groupby(["Budget Line Code", "match_label"])
            .size()
            .reset_index(name="Nombre")
        )
        fig = px.bar(
            mismatch_grp,
            x="Budget Line Code",
            y="Nombre",
            color="match_label",
            color_discrete_map={"Discordance": "#ff006e", "Concordance": "#3a86ff"},
            title="Taux de discordance des descriptions par ligne budgétaire",
            labels={"match_label": ""},
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "📂 Rapport Financier · Variables : `Budget Line Code`, `match_description` (RapidFuzz ≥ 90) · "
            "Une barre rouge importante indique que le champ description ne correspond pas au bénéficiaire."
        )

    fig = px.scatter(
        fr_filtered,
        x="Transaction Date",
        y="Total Amount In USD",
        color="Velocity Score",
        size="Transactions per day",
        color_continuous_scale="Reds",
        hover_data=["Description", "Budget Line Code", "is_outlier_label"],
        title="Chronologie des transactions — Taille : nb tx/jour · Couleur : Velocity Score",
        labels={"Total Amount In USD": "Montant (USD)"},
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "📂 Rapport Financier · Variables : `Transaction Date`, `Total Amount In USD`, `Velocity Score`, `Transactions per day` · "
        "Les bulles grandes et foncées signalent des transactions nombreuses et de montant élevé concentrées sur un même jour."
    )

    missing_inv = (
        fr_filtered[fr_filtered["invoice_missing"] == 1]
        .groupby("Budget Line Code")
        .size()
        .reset_index(name="Factures manquantes")
        .sort_values("Factures manquantes", ascending=False)
    )
    if not missing_inv.empty:
        fig = px.bar(
            missing_inv,
            x="Budget Line Code",
            y="Factures manquantes",
            title="Dates de facture manquantes par ligne budgétaire",
            color_discrete_sequence=["#ff006e"],
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "📂 Rapport Financier · Variable : `Invoice Date` (nulle = manquante) · "
            "Toute transaction sans date de facture est un justificatif manquant — signal d'audit direct."
        )
    else:
        st.info("Aucune date de facture manquante dans le filtre actuel.")

    with st.expander("Matrice de corrélation (colonnes numériques)"):
        num_cols = ["Total Amount In USD", "Velocity Score", "Transactions per day",
                    "Relative deviation", "is_outlier", "match_description", "invoice_missing"]
        num_cols = [c for c in num_cols if c in fr_filtered.columns]
        corr = fr_filtered[num_cols].corr().round(2)
        fig = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            title="Matrice de corrélation — Rapport Financier",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "📂 Rapport Financier · Toutes les variables numériques · "
            "Une forte corrélation entre deux signaux (ex. `is_outlier` et `invoice_missing`) indique qu'ils se cumulent sur les mêmes transactions."
        )

# ONGLET 3 — Transactions SENED

with tab3:
    st.subheader("Transactions SENED — Signaux de fraude")
    st.info(
        "**Source :** `Updated_SENED_Budget_Solidarity_Syria-Turkey_2023_05_17_EN.xlsx` — "
        "Transactions du projet Solidarité Syrie-Turquie (~270 lignes). "
        "Les codes budget (`1.1.xx` RH, `2.x` déplacements, `3.1.x` matériels, `4.5` gestion de cas) "
        "correspondent à des enveloppes prédéfinies dans le budget projet. "
        "Deux signaux ont été calculés : "
        "les **valeurs aberrantes** (IQR sur le montant) "
        "et les **REF en double** (une référence comptable doit être unique — tout doublon est un double paiement potentiel). "
        "Le budget `4.5` et deux bénéficiaires (Safwan, Ali Al Sed) concentrent une part anormalement élevée des fonds."
    )

    s_col1, s_col2 = st.columns(2)
    with s_col1:
        budget_opts = sorted(sened["budget"].dropna().unique().tolist())
        sel_budgets = st.multiselect(
            "Filtrer par budget",
            options=budget_opts,
            default=budget_opts,
            key="sened_budget_filter",
        )
    with s_col2:
        year_opts = sorted(sened["year"].dropna().unique().tolist()) if "year" in sened.columns else []
        sel_years = st.multiselect(
            "Filtrer par année",
            options=[int(y) for y in year_opts],
            default=[int(y) for y in year_opts],
            key="sened_year_filter",
        )

    sened_f = sened[sened["budget"].isin(sel_budgets)].copy()
    if sel_years and "year" in sened_f.columns:
        sened_f = sened_f[sened_f["year"].isin(sel_years)]

    sened_f["is_outlier_label"] = sened_f["is_outlier"].map({0: "Normal", 1: "Aberrant"})

    c1, c2 = st.columns(2)

    with c1:
        if "recipient" in sened_f.columns:
            top_rec = (
                sened_f.groupby("recipient")["amount"]
                .sum()
                .nlargest(15)
                .reset_index()
                .sort_values("amount")
            )
            fig = px.bar(
                top_rec,
                x="amount",
                y="recipient",
                orientation="h",
                title="Top 15 bénéficiaires par montant total",
                labels={"amount": "Montant total", "recipient": "Bénéficiaire"},
                color_discrete_sequence=["#3a86ff"],
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "📂 SENED · Variables : `recipient`, `amount` · "
                "Une concentration excessive sur un ou deux bénéficiaires est un signal de favoritisme ou de détournement."
            )

    with c2:
        if "currency" in sened_f.columns:
            budget_curr = (
                sened_f.groupby(["budget", "currency"])["amount"]
                .sum()
                .reset_index()
            )
            fig = px.bar(
                budget_curr,
                x="budget",
                y="amount",
                color="currency",
                title="Dépenses par ligne budgétaire et devise",
                labels={"amount": "Montant total", "budget": "Code budget"},
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "📂 SENED · Variables : `budget`, `amount`, `currency` · "
                "Une devise inattendue sur une ligne budgétaire (ex. USD pour des charges locales) peut indiquer une mauvaise imputation."
            )

    c3, c4 = st.columns(2)

    with c3:
        if "ref" in sened_f.columns:
            dup_refs = (
                sened_f[sened_f["ref_duplicate"] == 1]
                .groupby("ref")
                .size()
                .reset_index(name="Occurrences")
                .sort_values("Occurrences", ascending=False)
                .head(20)
                .sort_values("Occurrences")
            )
            if not dup_refs.empty:
                fig = px.bar(
                    dup_refs,
                    x="Occurrences",
                    y="ref",
                    orientation="h",
                    title="Références comptables en double (signal de fraude)",
                    labels={"Occurrences": "Nb d'occurrences", "ref": "REF"},
                    color_discrete_sequence=["#ff006e"],
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption(
                    "📂 SENED · Variable : `ref` · "
                    "Chaque REF doit correspondre à une seule opération — toute répétition indique un double paiement potentiel."
                )
            else:
                st.info("Aucun REF en double dans le filtre actuel.")

    with c4:
        if "period" in sened_f.columns:
            hover_cols = [c for c in ["recipient", "reason", "ref"] if c in sened_f.columns]
            fig = px.scatter(
                sened_f,
                x="period",
                y="amount",
                color="is_outlier_label",
                color_discrete_map={"Normal": "#3a86ff", "Aberrant": "#ff006e"},
                hover_data=hover_cols,
                title="Montants aberrants par période",
                labels={"is_outlier_label": "", "amount": "Montant", "period": "Période"},
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "📂 SENED · Variables : `period`, `amount`, `is_outlier` (méthode IQR) · "
                "Les points rouges sont des montants statistiquement atypiques — survoler pour voir le bénéficiaire et la raison."
            )

# ONGLET 4 — Transactions Suspectes

with tab4:
    st.subheader("Transactions suspectes — Lignes à risque élevé")
    st.info(
        "Ce tableau regroupe toutes les lignes ayant déclenché **au moins un signal d'anomalie**. "
        "La colonne `flags` liste les signaux actifs sur chaque ligne. "
        "**Les lignes cumulant plusieurs flags sont les priorités d'audit les plus élevées.** "
        "Signaux disponibles — Rapport Financier : `high_velocity` (Velocity Score > 75e pct), "
        "`mismatch` (description ≠ bénéficiaire), `outlier` (montant aberrant IQR), `invoice_missing` (facture absente). "
        "Signaux SENED : `outlier` (montant aberrant IQR), `ref_duplicate` (REF utilisée plusieurs fois). "
        "Le bouton **Exporter en CSV** permet de télécharger la sélection pour la transmettre aux auditeurs."
    )

    dataset_choice = st.radio(
        "Jeu de données",
        ["Rapport Financier", "Transactions SENED"],
        horizontal=True,
    )

    if dataset_choice == "Rapport Financier":
        flagged_df = get_finance_flags(fr)
        all_flags = ["high_velocity", "mismatch", "outlier", "invoice_missing"]
        display_cols = [
            "Transaction Date", "Budget Line Code", "Description",
            "2nd Description (Recipient)", "Total Amount In USD",
            "Velocity Score", "match_description", "is_outlier",
            "invoice_missing", "flags",
        ]
    else:
        flagged_df = get_sened_flags(sened)
        all_flags = ["outlier", "ref_duplicate"]
        display_cols = [c for c in ["date_voucher", "budget", "ref", "recipient",
                                     "reason", "amount", "amount_eur",
                                     "is_outlier", "ref_duplicate", "flags"]
                        if c in flagged_df.columns]

    selected_flags = st.multiselect(
        "Filtrer par type de signal",
        options=all_flags,
        default=all_flags,
        key="flag_filter",
    )

    if selected_flags:
        mask = flagged_df["flags"].apply(
            lambda f: any(flag in f for flag in selected_flags)
        )
        display_df = flagged_df[mask]
    else:
        display_df = flagged_df

    display_cols = [c for c in display_cols if c in display_df.columns]

    st.write(f"**{len(display_df)} transaction(s) suspecte(s)**")
    st.dataframe(display_df[display_cols], use_container_width=True, height=400)

    csv = display_df[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Exporter en CSV",
        data=csv,
        file_name=f"suspects_{dataset_choice.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )
