# LLM Job Matcher

Pipeline de matching CV / offres d'emploi avec un LLM local (LM Studio), sur deux échelles :

| Échelle | Source | Langue | Accès |
|---|---|---|---|
| Internationale | [LinkedIn Job Postings 2023-2024](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings) (Kaggle) | anglais | CSV à télécharger |
| Nationale (France) | [API Offres d'emploi v2](https://www.data.gouv.fr/dataservices/api-offres-demploi) de France Travail | français | API gratuite, compte francetravail.io |

Les deux sources sont converties dans un **schéma commun** (table `offres`) : identifiant préfixé
(`li_…` / `ft_…`), échelle, langue, titre, description, entreprise, pays, région, ville, type de
contrat, niveau d'expérience, salaire annuel, devise, date, URL, code ROME, compétences fournies par la source.

## Installation

```bash
pip install -r requirements.txt
```

## Données

**LinkedIn** : placer `postings.csv` dans `data/raw/` (et optionnellement `jobs/job_skills.csv`,
`mappings/skills.csv`).

**France Travail** :
1. Créer un compte sur [francetravail.io](https://francetravail.io), déclarer une application,
   l'abonner à l'API « Offres d'emploi v2 ».
2. Copier `.env.example` en `.env` et y mettre `FT_CLIENT_ID` et `FT_CLIENT_SECRET`
   (`.env` est ignoré par Git).
3. Les mots-clés de recherche sont dans `src/config.py` (`FT_MOTS_CLES`). L'API renvoie au plus
   1 150 offres par requête : on fait une requête par mot-clé, puis on dédoublonne.

Chaque collecte est sauvegardée telle quelle dans `data/raw/france_travail/offres_AAAA-MM-JJ.json`.

**CV** : PDF dans `data/raw/cv/`.

## Exécution

```bash
python -m src.ingestion --source linkedin                    # échelle internationale
python -m src.ingestion --source france_travail              # collecte via l'API
python -m src.ingestion --source france_travail --hors-ligne # retraite les JSON déjà collectés
python -m src.ingestion --source cv                          # CV seuls
python -m src.ingestion                                      # tout
python -m pytest -q                                          # tests (données simulées)
```

> Si une ancienne base `data/processed/job_matcher.db` existe (schéma LinkedIn seul),
> la supprimer avant la première ingestion.

## Structure

| Dossier / fichier | Rôle |
|---|---|
| `src/config.py` | chemins, LM Studio, filtres, paramètres France Travail |
| `src/sources/linkedin.py` | chargement + filtrage LinkedIn vers le schéma commun |
| `src/sources/france_travail.py` | OAuth2, pagination, parsing salaire / région / expérience |
| `src/ingestion.py` | orchestration des sources + extraction texte des CV |
| `src/storage.py` | base SQLite (offres, extractions, profils, matchings, suivi) |
| `src/llm_client.py` | appels au serveur LM Studio |
| `src/extraction.py` | extraction structurée (offres + CV) |
| `src/scoring.py` | score de matching |
| `src/aggregation.py` | métriques, gap analysis, comparaison France / international |
| `dashboard/app.py` | dashboard Streamlit |
| `tests/` | tests de la couche données |
| `eval/` | échantillon annoté + métriques |

## Mentions

Offres nationales : source France Travail, API Offres d'emploi v2, date de collecte indiquée
dans le nom du fichier JSON.
