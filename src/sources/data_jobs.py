"""Échelle nationale (solution de remplacement) : dataset « data_jobs » de Luke Barousse.

Pourquoi : l'API Offres d'emploi de France Travail est passée en accès sous conditions
(octobre 2026). Une demande d'accès a été envoyée le 15/11/2026 ; en attendant, on utilise
les offres françaises de data_jobs (Hugging Face, licence Apache-2.0, offres data de 2023).

Particularité : pas de description complète, mais des compétences déjà extraites (job_skills).
Ces offres n'ont donc pas besoin du LLM : l'étape d'extraction reprend directement ces compétences.

Téléchargement (une seule fois, depuis le PC) :
    python -m src.sources.data_jobs --telecharger
"""
import argparse
import ast
import hashlib
import re

import pandas as pd

from src.config import RAW_DIR, SEED, TAILLE_ECHANTILLON

FICHIER = RAW_DIR / "data_jobs.parquet"
HF_DATASET = "lukebarousse/data_jobs"

# Postes data comparables à l'échantillon LinkedIn (hors postes seniors et hors métiers non data)
POSTES_RETENUS = {
    "Data Analyst", "Data Scientist", "Data Engineer", "Business Analyst", "Machine Learning Engineer",
}

VILLES_REGIONS = {
    "paris": "Île-de-France", "boulogne-billancourt": "Île-de-France", "nanterre": "Île-de-France",
    "courbevoie": "Île-de-France", "puteaux": "Île-de-France", "issy-les-moulineaux": "Île-de-France",
    "levallois-perret": "Île-de-France", "saint-denis": "Île-de-France", "la défense": "Île-de-France",
    "neuilly-sur-seine": "Île-de-France", "massy": "Île-de-France", "versailles": "Île-de-France",
    "île-de-france": "Île-de-France", "ile-de-france": "Île-de-France",
    "lyon": "Auvergne-Rhône-Alpes", "grenoble": "Auvergne-Rhône-Alpes", "villeurbanne": "Auvergne-Rhône-Alpes",
    "clermont-ferrand": "Auvergne-Rhône-Alpes", "saint-étienne": "Auvergne-Rhône-Alpes", "annecy": "Auvergne-Rhône-Alpes",
    "toulouse": "Occitanie", "montpellier": "Occitanie",
    "bordeaux": "Nouvelle-Aquitaine", "pessac": "Nouvelle-Aquitaine", "limoges": "Nouvelle-Aquitaine",
    "nantes": "Pays de la Loire", "angers": "Pays de la Loire", "le mans": "Pays de la Loire",
    "lille": "Hauts-de-France", "amiens": "Hauts-de-France", "villeneuve-d'ascq": "Hauts-de-France",
    "marseille": "Provence-Alpes-Côte d'Azur", "nice": "Provence-Alpes-Côte d'Azur",
    "aix-en-provence": "Provence-Alpes-Côte d'Azur", "sophia antipolis": "Provence-Alpes-Côte d'Azur",
    "valbonne": "Provence-Alpes-Côte d'Azur", "toulon": "Provence-Alpes-Côte d'Azur",
    "strasbourg": "Grand Est", "nancy": "Grand Est", "metz": "Grand Est", "reims": "Grand Est", "mulhouse": "Grand Est",
    "rennes": "Bretagne", "brest": "Bretagne",
    "rouen": "Normandie", "caen": "Normandie", "le havre": "Normandie",
    "dijon": "Bourgogne-Franche-Comté", "besançon": "Bourgogne-Franche-Comté",
    "montbéliard": "Bourgogne-Franche-Comté", "belfort": "Bourgogne-Franche-Comté",
    "tours": "Centre-Val de Loire", "orléans": "Centre-Val de Loire",
}


def telecharger():
    """Télécharge le dataset depuis Hugging Face et le sauvegarde en Parquet (pip install datasets)."""
    from datasets import load_dataset

    df = load_dataset(HF_DATASET, split="train").to_pandas()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FICHIER, index=False)
    print(f"[data_jobs] {len(df)} offres sauvegardées dans {FICHIER}")
    print(df["job_country"].value_counts().head(10).to_string())


def _liste(v) -> list[str] | None:
    """job_skills arrive sous forme de texte \"['python', 'sql']\" (CSV) ou de liste (Parquet)."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, str):
        try:
            v = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            return None
    try:
        out = [str(s).strip().lower() for s in v if str(s).strip()]
    except TypeError:
        return None
    return out or None


def lieu(location: str | None) -> tuple[str | None, str | None]:
    """'Lyon, France' -> ('Auvergne-Rhône-Alpes', 'Lyon')."""
    if not isinstance(location, str):
        return None, None
    ville = location.split(",")[0].strip()
    if ville.lower() in ("france", "anywhere", ""):
        return None, None
    return VILLES_REGIONS.get(ville.lower()), ville


def contrat(schedule: str | None) -> str:
    s = schedule.lower() if isinstance(schedule, str) else ""   # valeur vide (NaN) dans le dataset
    if "intern" in s:
        return "Stage"
    if "contract" in s:
        return "Freelance"
    if "temp" in s:
        return "Intérim"
    if "full-time" in s:
        return "CDI"
    return "Autre"


def niveau(titre: str | None, type_contrat: str) -> str | None:
    t = titre.lower() if isinstance(titre, str) else ""
    if type_contrat == "Stage" or re.search(r"\bstag|intern", t):
        return "Stage"
    if re.search(r"alternan|apprenti", t):
        return "Débutant"
    if re.search(r"\bjunior\b|\bjr\b|débutant", t):
        return "Junior"
    if re.search(r"senior|\bsr\b|lead|principal|expert|confirm", t):
        return "Senior"
    return None


def _id(row) -> str:
    cle = f"{row.get('job_title')}|{row.get('company_name')}|{row.get('job_location')}|{row.get('job_posted_date')}"
    return "dj_" + hashlib.md5(cle.encode("utf-8")).hexdigest()[:12]


def charger(taille: int = TAILLE_ECHANTILLON, pays: str = "France") -> pd.DataFrame:
    if FICHIER.exists():
        df = pd.read_parquet(FICHIER)
    elif FICHIER.with_suffix(".csv").exists():
        df = pd.read_csv(FICHIER.with_suffix(".csv"))
    else:
        raise FileNotFoundError(
            f"{FICHIER} introuvable : lance d'abord  python -m src.sources.data_jobs --telecharger"
        )
    total = len(df)
    df = df[(df["job_country"] == pays) & (df["job_title_short"].isin(POSTES_RETENUS))].copy()
    df["competences"] = df["job_skills"].apply(_liste)
    df = df.dropna(subset=["job_title", "competences"])
    df["offre_id"] = df.apply(_id, axis=1)
    df = df.drop_duplicates("offre_id")
    retenues = len(df)
    if len(df) > taille:
        df = df.sample(n=taille, random_state=SEED)
    print(f"[data_jobs] {total} offres lues, {retenues} postes data en {pays}, {len(df)} gardées")

    lieux = df["job_location"].apply(lieu)
    contrats = df["job_schedule_type"].apply(contrat)
    out = pd.DataFrame({
        "offre_id": df["offre_id"],
        "source": "data_jobs",
        "echelle": "nationale",
        "langue": "fr",
        "titre": df["job_title"],
        # Pas de description complète dans ce dataset : texte court reconstruit, utilisé par les embeddings
        "description": df["job_title"] + ". Compétences : " + df["competences"].apply(", ".join) + ".",
        "entreprise": df["company_name"],
        "pays": pays,
        "region": lieux.str[0],
        "ville": lieux.str[1],
        "type_contrat": contrats,
        "niveau_experience": [niveau(t, c) for t, c in zip(df["job_title"], contrats)],
        "teletravail": df["job_work_from_home"].fillna(False).astype(int),
        "salaire_annuel": df["salary_year_avg"],
        "devise": "USD",   # data_jobs exprime les salaires en dollars
        "date_publication": pd.to_datetime(df["job_posted_date"]).dt.strftime("%Y-%m-%d"),
        "url": None,
        "code_rome": None,
        "competences_source": df["competences"],
    })
    return out.reset_index(drop=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--telecharger", action="store_true", help="Télécharge data_jobs depuis Hugging Face")
    if p.parse_args().telecharger:
        telecharger()
