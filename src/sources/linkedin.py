"""Échelle internationale : dataset Kaggle « LinkedIn Job Postings 2023-2024 ».

Fichiers attendus dans data/raw/ :
    postings.csv                    (obligatoire)
    jobs/job_skills.csv             (optionnel : catégories de compétences)
    mappings/skills.csv             (optionnel : libellés des catégories)
"""
import re

import pandas as pd

from src.config import (
    NIVEAUX_CIBLES_LINKEDIN, RAW_DIR, SEED, TAILLE_ECHANTILLON, TITRES_CIBLES_EN,
)

FACTEUR_ANNUEL = {"HOURLY": 2080, "WEEKLY": 52, "BIWEEKLY": 26, "MONTHLY": 12, "YEARLY": 1}

NIVEAUX = {
    "Internship": "Stage",
    "Entry level": "Débutant",
    "Associate": "Junior",
    "Mid-Senior level": "Confirmé",
    "Director": "Senior",
    "Executive": "Senior",
}
CONTRATS = {
    "Full-time": "CDI",          # approximation : LinkedIn ne distingue pas CDI / CDD
    "Contract": "Freelance",
    "Part-time": "Autre",
    "Temporary": "CDD",
    "Internship": "Stage",
    "Volunteer": "Autre",
    "Other": "Autre",
}


def salaire_annuel(row) -> float | None:
    """normalized_salary si présent, sinon médian ou moyenne min/max annualisé via pay_period."""
    if pd.notna(row.get("normalized_salary")):
        return float(row["normalized_salary"])
    facteur = FACTEUR_ANNUEL.get(str(row.get("pay_period", "")).upper())
    if not facteur:
        return None
    if pd.notna(row.get("med_salary")):
        base = row["med_salary"]
    elif pd.notna(row.get("min_salary")) and pd.notna(row.get("max_salary")):
        base = (row["min_salary"] + row["max_salary"]) / 2
    else:
        return None
    return float(base) * facteur


def decouper_lieu(location: str) -> tuple[str, str | None, str | None]:
    """'Austin, TX' -> ('United States', 'TX', 'Austin'). 'United States' -> (pays, None, None)."""
    if not isinstance(location, str) or not location.strip():
        return "United States", None, None
    m = re.match(r"^\s*(.+?),\s*([A-Z]{2})\s*$", location)
    if m:
        return "United States", m.group(2), m.group(1)
    return "United States", None, None


def _categories(postings: pd.DataFrame) -> pd.Series:
    js, sk = RAW_DIR / "jobs" / "job_skills.csv", RAW_DIR / "mappings" / "skills.csv"
    if not (js.exists() and sk.exists()):
        return pd.Series([None] * len(postings), index=postings.index)
    lien = pd.read_csv(js).merge(pd.read_csv(sk), on="skill_abr", how="left")
    par_offre = lien.groupby("job_id")["skill_name"].apply(lambda s: sorted(set(s.dropna())))
    return postings["job_id"].map(par_offre)


def charger(taille: int = TAILLE_ECHANTILLON) -> pd.DataFrame:
    chemin = RAW_DIR / "postings.csv"
    if not chemin.exists():
        raise FileNotFoundError(f"{chemin} introuvable : télécharge le dataset Kaggle dans data/raw/")
    df = pd.read_csv(chemin)
    total = len(df)

    df = df[df["title"].str.contains(TITRES_CIBLES_EN, case=False, na=False, regex=True)]
    df = df[df["formatted_experience_level"].isin(NIVEAUX_CIBLES_LINKEDIN)]
    df = df.dropna(subset=["description"])
    retenues = len(df)
    if len(df) > taille:
        df = df.sample(n=taille, random_state=SEED)
    print(f"[LinkedIn] {total} offres lues, {retenues} postes data ciblés, {len(df)} gardées")

    lieux = df["location"].apply(decouper_lieu)
    out = pd.DataFrame({
        "offre_id": "li_" + df["job_id"].astype(str),
        "source": "linkedin",
        "echelle": "internationale",
        "langue": "en",
        "titre": df["title"],
        "description": df["description"],
        "entreprise": df["company_name"],
        "pays": lieux.str[0],
        "region": lieux.str[1],
        "ville": lieux.str[2],
        "type_contrat": df["formatted_work_type"].map(CONTRATS).fillna("Autre"),
        "niveau_experience": df["formatted_experience_level"].map(NIVEAUX),
        "teletravail": df["remote_allowed"].fillna(0).astype(int),
        "salaire_annuel": df.apply(salaire_annuel, axis=1),
        "devise": df["currency"].fillna("USD"),
        "date_publication": pd.to_datetime(df["listed_time"], unit="ms").dt.strftime("%Y-%m-%d"),
        "url": df["job_posting_url"],
        "code_rome": None,
        "competences_source": _categories(df),
    })
    return out.reset_index(drop=True)
