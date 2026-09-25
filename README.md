# LLM Job Matcher

Pipeline de matching CV / offres d'emploi avec un LLM local (LM Studio).

## Installation
```bash
pip install -r requirements.txt
```

## Données
- Dataset : [LinkedIn Job Postings 2023-2024](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings)
- Placer `postings.csv` dans `data/raw/` (et optionnellement `jobs/` et `mappings/`)
- Placer le(s) CV PDF dans `data/raw/cv/`

## Exécution
```bash
python -m src.ingestion          # 1. chargement + filtrage + SQLite
```

## Structure
| Dossier / fichier | Rôle |
|---|---|
| `src/config.py` | chemins, modèle LM Studio, filtres |
| `src/ingestion.py` | chargement CSV + extraction texte des CV |
| `src/llm_client.py` | appels au serveur LM Studio |
| `src/extraction.py` | extraction structurée (offres + CV) |
| `src/scoring.py` | score de matching |
| `src/storage.py` | base SQLite |
| `src/aggregation.py` | métriques, gap analysis |
| `dashboard/app.py` | dashboard Streamlit |
| `eval/` | échantillon annoté + métriques de précision |
| `models/` | notes sur les modèles testés |
