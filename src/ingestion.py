"""Étape 1 — Ingestion des deux échelles + des CV, vers la base SQLite.

Exemples (depuis la racine du projet) :
    python -m src.ingestion --source linkedin              # échelle internationale
    python -m src.ingestion --source france_travail        # appelle l'API France Travail
    python -m src.ingestion --source france_travail --hors-ligne   # retraite les JSON déjà collectés
    python -m src.ingestion --source all                   # les deux + les CV
    python -m src.ingestion --source cv                    # CV seuls
"""
import argparse
from datetime import datetime

import pdfplumber

from src import storage
from src.config import CV_DIR, PROCESSED_DIR
from src.sources import france_travail, linkedin


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


def ingerer(source: str, hors_ligne: bool = False):
    storage.init_db()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if source in ("linkedin", "all"):
        df = linkedin.charger()
        storage.enregistrer_offres(df)
        df.to_csv(PROCESSED_DIR / "offres_internationales.csv", index=False)

    if source in ("france_travail", "all"):
        df = france_travail.charger(appeler_api=not hors_ligne)
        storage.enregistrer_offres(df)
        df.to_csv(PROCESSED_DIR / "offres_nationales.csv", index=False)

    if source in ("cv", "all"):
        ingerer_cv()

    print("\n=== Contenu de la base ===")
    print(storage.resume_offres().to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Ingestion des offres (deux échelles) et des CV")
    p.add_argument("--source", choices=["linkedin", "france_travail", "cv", "all"], default="all")
    p.add_argument("--hors-ligne", action="store_true",
                   help="France Travail : réutiliser les JSON déjà collectés sans appeler l'API")
    args = p.parse_args()
    ingerer(args.source, args.hors_ligne)
