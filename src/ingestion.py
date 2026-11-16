"""Étape 1 — Ingestion des deux échelles + des CV, vers la base SQLite.

Exemples (depuis la racine du projet) :
    python -m src.ingestion                                # LinkedIn + source nationale + CV
    python -m src.ingestion --source linkedin              # échelle internationale
    python -m src.ingestion --source france_travail        # échelle nationale : appelle l'API
    python -m src.ingestion --source france_travail --hors-ligne   # retraite les JSON déjà collectés
    python -m src.ingestion --source data_jobs             # échelle nationale de secours
    python -m src.ingestion --source cv                    # CV seuls
    python -m src.ingestion --purger data_jobs             # retire une source de la base

La source nationale utilisée par défaut est SOURCE_NATIONALE (src/config.py), France Travail.
Une seule source nationale doit être en base à la fois : purger l'autre avant de changer.
"""
import argparse
from datetime import datetime

import pdfplumber

from src import storage
from src.config import CV_DIR, PROCESSED_DIR, SOURCE_NATIONALE
from src.sources import data_jobs, france_travail, linkedin


def ingerer_cv() -> int:
    n = 0
    for pdf in sorted(CV_DIR.glob("*.pdf")):
        with pdfplumber.open(pdf) as doc:
            texte = "\n".join(page.extract_text() or "" for page in doc.pages).strip()
        storage.enregistrer_profil({
            "profil_id": pdf.stem,
            "texte_brut": texte,
            "date_extraction": datetime.now().isoformat(timespec="seconds"),
        })
        n += 1
        print(f"[CV] {pdf.name} : {len(texte)} caractères")
    if n == 0:
        print(f"[CV] aucun PDF dans {CV_DIR}")
    return n


def _verifier_une_source_nationale(source: str):
    autres = storage.lire("SELECT DISTINCT source FROM offres WHERE echelle = 'nationale' AND source != ?",
                          (source,))["source"].tolist()
    if autres:
        raise RuntimeError(f"La base contient déjà des offres nationales de {autres}. "
                           f"Lance d'abord : python -m src.ingestion --purger {autres[0]}")


def ingerer(source: str, hors_ligne: bool = False):
    storage.init_db()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    nationale = SOURCE_NATIONALE if source == "all" else source

    if source in ("linkedin", "all"):
        df = linkedin.charger()
        storage.enregistrer_offres(df)
        df.to_csv(PROCESSED_DIR / "offres_internationales.csv", index=False)

    if nationale == "france_travail":
        _verifier_une_source_nationale("france_travail")
        df = france_travail.charger(appeler_api=not hors_ligne)
        storage.enregistrer_offres(df)
        df.to_csv(PROCESSED_DIR / "offres_nationales.csv", index=False)

    if nationale == "data_jobs":
        _verifier_une_source_nationale("data_jobs")
        df = data_jobs.charger()
        storage.enregistrer_offres(df)
        df.to_csv(PROCESSED_DIR / "offres_nationales.csv", index=False)

    if source in ("cv", "all"):
        ingerer_cv()

    print("\n=== Contenu de la base ===")
    print(storage.resume_offres().to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Ingestion des offres (deux échelles) et des CV")
    p.add_argument("--source", choices=["linkedin", "data_jobs", "france_travail", "cv", "all"], default="all")
    p.add_argument("--hors-ligne", action="store_true",
                   help="France Travail : réutiliser les JSON déjà collectés sans appeler l'API")
    p.add_argument("--purger", choices=["linkedin", "data_jobs", "france_travail"],
                   help="Supprime de la base les offres de cette source (et leurs extractions et scores)")
    args = p.parse_args()
    if args.purger:
        storage.init_db()
        print(f"{storage.supprimer_source(args.purger)} offres {args.purger} supprimées")
    else:
        ingerer(args.source, args.hors_ligne)
