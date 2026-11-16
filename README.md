# LLM Job Matcher

Pipeline de matching CV / offres d'emploi avec un LLM local (LM Studio), sur deux échelles.

| Échelle | Source | Contenu | Rôle du LLM |
|---|---|---|---|
| Internationale | [LinkedIn Job Postings 2023-2024](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings) (Kaggle) | descriptions complètes, en anglais | extrait compétences, niveau, contrat, résumé |
| Nationale (France) | [data_jobs](https://huggingface.co/datasets/lukebarousse/data_jobs) (Hugging Face, Apache-2.0), offres France 2023 | titres + compétences déjà extraites | aucun (compétences reprises de la source) |

L'[API Offres d'emploi de France Travail](https://www.data.gouv.fr/dataservices/api-offres-demploi),
prévue au départ pour l'échelle nationale, est passée en accès sous conditions en octobre 2026.
Une demande d'accès a été envoyée le 15/11/2026 ; le connecteur `src/sources/france_travail.py`
est prêt et testé pour le jour où l'accès est accordé.

> Aucun modèle n'est entraîné : le LLM (Qwen2.5-7B-Instruct) est utilisé tel quel avec un prompt
> et une sortie JSON validée ; les embeddings (multilingual-e5-base) sont un modèle pré-entraîné.

## Pipeline

1. **Ingestion** (`src/ingestion.py`, `src/sources/`) : chargement, filtrage des postes data, schéma commun, SQLite.
2. **Extraction** (`src/extraction.py`, `src/llm_client.py`) : JSON structuré par le LLM local, normalisation des compétences.
3. **Scoring** (`src/scoring.py`) : 3 méthodes comparées — règles (part des compétences possédées), embeddings
   (similarité cosinus), jugement du LLM (optionnel, sur les meilleures offres). Score final = 0,6 règles + 0,4 embeddings.
4. **Agrégation** (`src/aggregation.py`) : compétences les plus demandées par échelle, écart profil / marché, salaires.
5. **Dashboard** (`dashboard/app.py`) : Streamlit.

## Installation

```bash
pip install -r requirements.txt
```

## Exécution, dans l'ordre

```bash
# 0. Données (une seule fois)
#    - postings.csv (+ jobs/job_skills.csv, mappings/skills.csv) de Kaggle dans data/raw/
#    - CV en PDF dans data/raw/cv/
python -m src.sources.data_jobs --telecharger     # télécharge data_jobs (~230 Mo)

# 1. Ingestion
python -m src.ingestion                           # LinkedIn + data_jobs (France) + CV

# 2. Extraction (LM Studio lancé, serveur démarré)
python -m src.extraction --test                   # vérifie que LM Studio répond
python -m src.extraction --offres --echelle nationale            # instantané (sans LLM)
python -m src.extraction --offres --echelle internationale --limite 20   # essai
python -m src.extraction --offres --echelle internationale --limite 200  # relançable
python -m src.extraction --cv

# 3. Scoring
python -m src.scoring                             # rapide
python -m src.scoring --llm-top 10                # + avis du LLM sur 10 offres par échelle

# 4. Dashboard
streamlit run dashboard/app.py

# Tests (données simulées, sans LM Studio)
python -m pytest -q
```

> Si une ancienne base `data/processed/job_matcher.db` existe, la supprimer avant la première ingestion.

## Limites connues

- Les deux échelles ne sont pas extraites de la même façon (LLM pour LinkedIn, compétences fournies par
  la source pour data_jobs) : une partie des écarts de compétences vient de cette différence de méthode.
- Salaires en USD dans les deux datasets, comparés seulement à l'intérieur d'une échelle.
- Offres datées (2023-2024).
