# LLM Job Matcher

Pipeline de matching CV / offres d'emploi avec un LLM local (LM Studio), sur deux échelles.

| Échelle | Source | Contenu | Rôle du LLM |
|---|---|---|---|
| Internationale | [LinkedIn Job Postings 2023-2024](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings) (Kaggle) | descriptions complètes, en anglais | extrait compétences, niveau, contrat, résumé |
| Nationale (France) | [API Offres d'emploi v2](https://www.data.gouv.fr/dataservices/api-offres-demploi) de France Travail, offres actives | descriptions complètes, en français | extrait compétences, niveau, contrat, résumé |
| Nationale, secours | [data_jobs](https://huggingface.co/datasets/lukebarousse/data_jobs) (Hugging Face, Apache-2.0), offres France 2023 | titres + compétences déjà extraites | aucun |

L'API France Travail est en accès sous conditions depuis octobre 2026. Accès demandé le 15/11/2026,
accordé le 16/11/2026 pour l'usage déclaré (projet universitaire, ~1 000 offres, conservation locale,
LLM exécuté en local). Toute évolution de cet usage doit être signalée à France Travail avant sa mise en œuvre.
Source des offres nationales : France Travail, avec la date d'extraction (nom du fichier JSON).

> Aucun modèle n'est entraîné : le LLM (Qwen2.5-7B-Instruct) est utilisé tel quel avec un prompt
> et une sortie JSON validée ; les embeddings (multilingual-e5-base) sont un modèle pré-entraîné.

## Pipeline

1. **Ingestion** (`src/ingestion.py`, `src/sources/`) : chargement, filtrage des postes data, schéma commun, SQLite.
2. **Extraction** (`src/extraction.py`, `src/llm_client.py`) : JSON structuré par le LLM local, normalisation des compétences.
3. **Scoring** (`src/scoring.py`) : 3 méthodes comparées — règles (part des compétences possédées), embeddings
   (similarité cosinus), jugement du LLM (optionnel, sur les meilleures offres). Score final = 0,6 règles + 0,4 embeddings,
   multiplié par une pénalité de niveau (offre Confirmé ×0,75 / Senior ×0,5 pour un profil junior, réglable via
   `PROFIL_NIVEAU`) et une pénalité d'incompatibilité (habilitation de sécurité, bénévolat ×0,3).
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
#    - identifiants France Travail dans .env (voir .env.example)

# 1. Ingestion
python -m src.ingestion                           # LinkedIn + France Travail + CV
python -m src.ingestion --purger data_jobs        # si data_jobs avait été ingéré avant

# 2. Extraction (LM Studio lancé, serveur démarré)
python -m src.extraction --test                   # vérifie que LM Studio répond
python -m src.extraction --offres --echelle nationale --limite 150      # relançable
python -m src.extraction --offres --echelle internationale --limite 20   # essai
python -m src.extraction --offres --echelle internationale --limite 200  # relançable
python -m src.extraction --cv

# 3. Scoring
python -m src.scoring                             # rapide
python -m src.scoring --llm-top 10                # + avis du LLM sur 10 offres par échelle

# 4. Évaluation
python -m src.evaluation --preparer               # crée eval/annotation_matching.csv et eval/controle_extraction.csv
#    -> remplir à la main (Excel) : pertinence 2/1/0 ; ok_competences / ok_niveau / ok_contrat 1/0
python -m src.evaluation --calculer               # précision@10, nDCG@10, Spearman, champs corrects, journal LLM
python -m src.evaluation --modeles qwen2.5-7b-instruct qwen2.5-3b-instruct --n 30   # comparaison LM Studio

# 5. Dashboard
streamlit run dashboard/app.py

# Tests (données simulées, sans LM Studio)
python -m pytest -q
```

> Si une ancienne base `data/processed/job_matcher.db` existe, la supprimer avant la première ingestion.

## Limites connues

- Les deux échelles n'ont pas la même date : LinkedIn 2023-2024, France Travail = offres actives à la collecte.
- Salaires en USD (LinkedIn) et en EUR (France Travail) : comparés seulement à l'intérieur d'une échelle.
- Avec la source de secours data_jobs, les compétences françaises ne passent pas par le LLM.
