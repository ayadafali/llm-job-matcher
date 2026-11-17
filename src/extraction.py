"""Étape 2 — Extraction structurée des offres et des CV avec le LLM local.

Offres LinkedIn (anglais) / France Travail (français) : le LLM lit la description et renvoie
un JSON validé (compétences, niveau, contrat, années d'expérience, télétravail, résumé).
Offres data_jobs : compétences déjà extraites par la source -> reprises telles quelles, sans LLM.

Exemples :
    python -m src.extraction --test                         # vérifie que LM Studio répond
    python -m src.extraction --offres --limite 20           # essai sur 20 offres
    python -m src.extraction --offres --echelle internationale --limite 200
    python -m src.extraction --offres --echelle nationale   # data_jobs : instantané, sans LLM
    python -m src.extraction --cv
Chaque offre est enregistrée dès qu'elle est traitée : on peut interrompre et relancer.
"""
import argparse
import json
import re
import time
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field

from src import llm_client, storage
from src.config import LLM_MODEL

TAILLE_MAX_TEXTE = 4000   # caractères envoyés au LLM (limite le temps de calcul)


# --- Schémas de sortie imposés au LLM --------------------------------------------------------

class OffreExtraite(BaseModel):
    competences: list[str] = Field(description="Compétences techniques et outils exigés, noms courts")
    niveau_experience: Optional[Literal["Stage", "Débutant", "Junior", "Confirmé", "Senior"]]
    annees_experience_min: Optional[int]
    type_contrat: Optional[Literal["CDI", "CDD", "Stage", "Alternance", "Freelance", "Intérim", "Autre"]]
    teletravail: Optional[bool]
    resume: str = Field(description="Résumé du poste en une ou deux phrases, en français")


class ProfilExtrait(BaseModel):
    competences: list[str]
    experiences: list[str] = Field(description="Une ligne par expérience : poste, entreprise, durée")
    formation: list[str]
    resume: str


PROMPT_OFFRE = """Tu es un assistant d'extraction d'informations pour des offres d'emploi.
L'offre peut être en anglais ou en français ; réponds toujours en français, en JSON uniquement.
Règles :
- competences : outils, langages, logiciels et méthodes techniques EXPLICITEMENT demandés
  (ex. Python, SQL, Power BI, Excel, Tableau, Spark, AWS, machine learning, statistiques).
  Noms courts et standards, 15 au maximum, pas de qualités humaines, n'invente rien.
- niveau_experience : Stage, Débutant (0-1 an), Junior (1-2 ans), Confirmé (3-5 ans), Senior (plus de 5 ans) ; null si non indiqué.
- annees_experience_min : nombre entier d'années minimum demandé, null si non indiqué.
- type_contrat : CDI pour un poste permanent à temps plein, Stage, Alternance, CDD, Freelance, Intérim ou Autre ; null si non indiqué.
- teletravail : true si le télétravail (remote, hybrid) est mentionné, false s'il est exclu, null sinon.
- resume : une ou deux phrases en français décrivant le poste."""

PROMPT_CV = """Tu es un assistant d'extraction d'informations pour des CV.
Réponds en français, en JSON uniquement.
- competences : outils, langages, logiciels et méthodes techniques du candidat (noms courts, 30 au maximum).
- experiences : une ligne par expérience (poste, entreprise, durée).
- formation : une ligne par diplôme ou formation.
- resume : profil du candidat en deux phrases."""


# --- Normalisation des compétences ----------------------------------------------------------

SYNONYMES = {
    "powerbi": "power bi", "microsoft power bi": "power bi", "ms power bi": "power bi",
    "ms excel": "excel", "microsoft excel": "excel", "advanced excel": "excel",
    "python3": "python", "python 3": "python",
    "ml": "machine learning", "apprentissage automatique": "machine learning",
    "deep-learning": "deep learning", "apprentissage profond": "deep learning",
    "sklearn": "scikit-learn", "scikit learn": "scikit-learn",
    "postgres": "postgresql", "ms sql server": "sql server", "mssql": "sql server", "t-sql": "sql server",
    "amazon web services": "aws", "google cloud": "gcp", "google cloud platform": "gcp",
    "microsoft azure": "azure", "apache spark": "spark", "pyspark": "spark",
    "apache airflow": "airflow", "apache kafka": "kafka",
    "statistics": "statistiques", "statistical analysis": "statistiques", "statistique": "statistiques",
    "data visualization": "data visualisation", "data viz": "data visualisation", "dataviz": "data visualisation",
    "visualisation de données": "data visualisation",
    "natural language processing": "nlp", "traitement du langage naturel": "nlp",
    "r programming": "r", "langage r": "r",
    "git/github": "git", "github": "git", "gitlab": "git",
    "google sheets": "sheets", "looker studio": "looker",
    "ms office": "office", "microsoft office": "office",
}


def normaliser(competence: str) -> str:
    c = re.sub(r"\s+", " ", competence.strip().lower()).strip(" .;:-")
    return SYNONYMES.get(c, c)


def normaliser_liste(competences) -> list[str]:
    vues, out = set(), []
    for c in competences or []:
        n = normaliser(str(c))
        if n and n not in vues and len(n) <= 40:
            vues.add(n)
            out.append(n)
    return out


# --- Offres -----------------------------------------------------------------------------------

def extraire_offre(offre: dict, modele: str | None = None) -> OffreExtraite:
    texte = f"Intitulé : {offre['titre']}\nEntreprise : {offre.get('entreprise') or ''}\n\n" \
            f"{(offre.get('description') or '')[:TAILLE_MAX_TEXTE]}"
    return llm_client.chat_json(PROMPT_OFFRE, texte, OffreExtraite, modele=modele)


def extraire_offres(echelle: str | None = None, limite: int | None = None) -> int:
    a_traiter = storage.offres_non_extraites(echelle, limite)
    n, erreurs, debut = 0, 0, time.time()
    for i, offre in enumerate(a_traiter.to_dict("records"), 1):
        if offre["source"] == "data_jobs":
            comp = json.loads(offre["competences_source"]) if offre["competences_source"] else []
            ligne = {
                "offre_id": offre["offre_id"],
                "competences_requises": normaliser_liste(comp),
                "niveau_exp": offre["niveau_experience"],
                "type_contrat": offre["type_contrat"],
                "localisation": offre["region"] or offre["ville"],
                "salaire_estime": offre["salaire_annuel"],
                "resume": None,
                "modele": "source:data_jobs",
            }
        else:
            llm_client.dernier_appel.clear()
            try:
                x = extraire_offre(offre)
            except Exception as e:   # une offre en échec ne bloque pas les autres
                erreurs += 1
                print(f"  ! {offre['offre_id']} : {e}")
                storage.journaliser({"offre_id": offre["offre_id"], "modele": LLM_MODEL, "succes": 0,
                                     "essais": llm_client.dernier_appel.get("essais"),
                                     "duree_s": llm_client.dernier_appel.get("duree_s"),
                                     "erreur": str(e)[:300]})
                continue
            storage.journaliser({"offre_id": offre["offre_id"], "modele": LLM_MODEL, "succes": 1,
                                 "essais": llm_client.dernier_appel.get("essais"),
                                 "duree_s": llm_client.dernier_appel.get("duree_s"),
                                 "nb_competences": len(x.competences)})
            ligne = {
                "offre_id": offre["offre_id"],
                "competences_requises": normaliser_liste(x.competences),
                "niveau_exp": x.niveau_experience or offre["niveau_experience"],
                "type_contrat": x.type_contrat or offre["type_contrat"],
                "localisation": offre["region"] or offre["ville"],
                "salaire_estime": offre["salaire_annuel"],
                "resume": x.resume,
                "modele": LLM_MODEL,
            }
        storage.enregistrer_extraction(ligne)
        n += 1
        if offre["source"] != "data_jobs" and (i % 5 == 0 or i == len(a_traiter)):
            ecoule = time.time() - debut
            print(f"  {i}/{len(a_traiter)} offres, {ecoule / i:.1f} s par offre, {erreurs} échec(s)")
    print(f"[Extraction] {n} offres enregistrées, {erreurs} en échec")
    return n


# --- CV ------------------------------------------------------------------------------------------

def extraire_cvs() -> int:
    n = 0
    for p in storage.profils_non_extraits().to_dict("records"):
        x = llm_client.chat_json(PROMPT_CV, (p["texte_brut"] or "")[:TAILLE_MAX_TEXTE * 2], ProfilExtrait)
        storage.enregistrer_profil({
            "profil_id": p["profil_id"],
            "competences": normaliser_liste(x.competences),
            "experiences": x.experiences,
            "formation": x.formation,
            "modele": LLM_MODEL,
            "date_extraction": datetime.now().isoformat(timespec="seconds"),
        })
        print(f"[CV] {p['profil_id']} : {len(x.competences)} compétences")
        n += 1
    return n


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Extraction structurée avec le LLM local")
    p.add_argument("--test", action="store_true", help="Vérifie la connexion à LM Studio")
    p.add_argument("--offres", action="store_true")
    p.add_argument("--cv", action="store_true")
    p.add_argument("--echelle", choices=["internationale", "nationale"])
    p.add_argument("--limite", type=int)
    a = p.parse_args()
    storage.init_db()
    if a.test:
        print("Modèles chargés dans LM Studio :", llm_client.modeles_disponibles())
    if a.offres:
        extraire_offres(a.echelle, a.limite)
    if a.cv:
        extraire_cvs()
