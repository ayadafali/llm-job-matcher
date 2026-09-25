"""
Étape 1 — Ingestion : chargement du dataset LinkedIn + extraction texte des CV PDF.

Fichiers attendus :
  data/raw/postings.csv
  data/raw/jobs/job_skills.csv      (optionnel)
  data/raw/mappings/skills.csv      (optionnel)
  data/raw/cv/*.pdf

Lancer depuis la racine : python -m src.ingestion
"""
import re

import pandas as pd
import pdfplumber

from src import config, storage

FACTEUR_ANNUEL = {"HOURLY": 2080, "WEEKLY": 52, "BIWEEKLY": 26, "MONTHLY": 12, "YEARLY": 1}

RENOMMAGE = {
    "formatted_work_type": "work_type",
    "formatted_experience_level": "experience_level",
    "job_posting_url": "url",
}
COLONNES_FINALES = [
    "job_id", "company_name", "title", "description", "location", "etat",
    "work_type", "experience_level", "remote_allowed", "salaire_annuel",
    "currency", "date_publication", "url", "categories_linkedin",
]


def salaire_annuel(row):
    """normalized_salary si présent, sinon médiane (ou moyenne min/max) annualisée via pay_period."""
    if pd.notna(row.get("normalized_salary")):
        return row["normalized_salary"]
    base = row.get("med_salary")
    if pd.isna(base) and pd.notna(row.get("min_salary")) and pd.notna(row.get("max_salary")):
        base = (row["min_salary"] + row["max_salary"]) / 2
    facteur = FACTEUR_ANNUEL.get(str(row.get("pay_period")).upper())
    return base * facteur if pd.notna(base) and facteur else None


def categories_linkedin():
    """Catégories de compétences LinkedIn (larges) par offre, si les fichiers sont présents."""
    f_jobs = config.RAW_DIR / "jobs" / "job_skills.csv"
    f_map = config.RAW_DIR / "mappings" / "skills.csv"
    if not (f_jobs.exists() and f_map.exists()):
        return None
    skills = pd.read_csv(f_jobs).merge(pd.read_csv(f_map), on="skill_abr", how="left")
    return (
        skills.groupby("job_id")["skill_name"]
        .apply(lambda s: ", ".join(s.dropna()))
        .rename("categories_linkedin")
    )


def charger_offres() -> pd.DataFrame:
    df = pd.read_csv(config.RAW_DIR / "postings.csv")
    df = df.rename(columns=RENOMMAGE)

    # Filtrage : postes data + niveau d'expérience
    df = df[df["title"].str.contains(config.TITRES_CIBLES, flags=re.IGNORECASE, na=False)]
    df = df[df["experience_level"].isin(config.NIVEAUX_CIBLES)]
    df = df.dropna(subset=["description"])
    df = df.drop_duplicates(subset=["title", "company_name", "description"])

    # Enrichissement
    df["salaire_annuel"] = df.apply(salaire_annuel, axis=1)
    df["etat"] = df["location"].str.extract(r",\s*([A-Z]{2})$", expand=False)
    df["date_publication"] = (
        pd.to_datetime(df["listed_time"], unit="ms", errors="coerce").dt.strftime("%Y-%m-%d")
    )
    df["remote_allowed"] = df.get("remote_allowed", pd.Series(dtype=float)).fillna(0).astype(int)

    cat = categories_linkedin()
    df = df.merge(cat, on="job_id", how="left") if cat is not None else df.assign(categories_linkedin=None)

    if len(df) > config.TAILLE_ECHANTILLON:
        df = df.sample(config.TAILLE_ECHANTILLON, random_state=config.SEED)

    return df.reindex(columns=COLONNES_FINALES)


def charger_cvs() -> pd.DataFrame:
    lignes = []
    for pdf_path in sorted(config.CV_DIR.glob("*.pdf")):
        with pdfplumber.open(pdf_path) as pdf:
            texte = "\n".join(page.extract_text() or "" for page in pdf.pages)
        lignes.append({"profil_id": pdf_path.stem, "texte_brut": texte})
    return pd.DataFrame(lignes, columns=["profil_id", "texte_brut"])


def main():
    storage.init_db()

    offres = charger_offres()
    offres.to_csv(config.PROCESSED_DIR / "offres_echantillon.csv", index=False)
    storage.enregistrer_offres(offres)

    cvs = charger_cvs()
    storage.enregistrer_profils(cvs)

    print(f"Offres retenues : {len(offres)}")
    print(offres["experience_level"].value_counts().to_string())
    print(f"Offres avec salaire : {offres['salaire_annuel'].notna().mean():.0%}")
    print(f"Offres avec État identifié : {offres['etat'].notna().mean():.0%}")
    print(f"CV chargés : {len(cvs)} {list(cvs['profil_id'])}")


if __name__ == "__main__":
    main()
