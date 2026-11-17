"""Test de bout en bout : ingestion (LinkedIn + data_jobs) -> extraction -> scoring -> agrégation.

Le LLM et le modèle d'embeddings sont simulés : aucun serveur LM Studio ni téléchargement requis.
"""
import json

import numpy as np
import pandas as pd
import pytest

from src import aggregation, extraction, ingestion, llm_client, scoring, storage
from src.sources import data_jobs as dj
from tests.test_ingestion import env  # noqa: F401  (fixture : LinkedIn simulé + chemins temporaires)


@pytest.fixture
def env_complet(env, monkeypatch):  # noqa: F811
    raw = env / "raw"
    monkeypatch.setattr(dj, "FICHIER", raw / "data_jobs.parquet")
    monkeypatch.setattr(ingestion, "SOURCE_NATIONALE", "data_jobs")
    pd.DataFrame({
        "job_title_short": ["Data Analyst", "Data Scientist", "Data Engineer", "Data Analyst", "Software Engineer", "Data Analyst"],
        "job_title": ["Data Analyst F/H", "Stage Data Scientist", "Data Engineer Senior", "Junior Data Analyst", "Dev Java", "Data Analyst"],
        "job_location": ["Paris, France", "Lyon, France", "Nantes, France", "France", "Paris, France", "Austin, TX"],
        "job_schedule_type": ["Full-time", "Full-time and Internship", None, "Contractor", "Full-time", "Full-time"],
        "job_work_from_home": [False, True, False, False, False, False],
        "job_posted_date": ["2023-05-01 10:00:00"] * 6,
        "job_country": ["France"] * 5 + ["United States"],
        "salary_year_avg": [60000.0, None, None, None, None, 90000.0],
        "company_name": ["A", "B", "C", "D", "E", "F"],
        "job_skills": ["['sql', 'python', 'power bi']", "['python', 'scikit-learn']", "['spark', 'sql', 'aws']",
                       "['excel', 'sql']", "['java']", "['sql']"],
    }).to_parquet(raw / "data_jobs.parquet", index=False)
    return env


class FauxLLM:
    def __call__(self, systeme, utilisateur, schema, **kwargs):
        llm_client.dernier_appel.update(essais=1, succes=True, duree_s=0.01, erreur=None)
        if schema is extraction.OffreExtraite:
            comp = ["Python", "SQL"] if "SQL" in utilisateur or "ML" in utilisateur else ["Power BI", "MS Excel"]
            return schema(competences=comp, niveau_experience=None if "Intern" in utilisateur else "Débutant",
                          annees_experience_min=0, type_contrat=None if "Sales" in utilisateur else "CDI",
                          teletravail=None, resume="Poste d'analyste de données.")
        if schema is extraction.ProfilExtrait:
            return schema(competences=["Python", "SQL", "Power BI"], experiences=["Stage data, 6 mois"],
                          formation=["Master 2"], resume="Profil data analyst.")
        return schema(score=70, justification="Bonne adéquation.")


def faux_encoder(textes, role):
    vocab = ["python", "sql", "power bi", "excel", "spark", "aws", "scikit-learn", "data"]
    v = np.array([[t.lower().count(m) + 0.1 for m in vocab] for t in textes], dtype=float)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_pipeline_complet(env_complet, monkeypatch):
    monkeypatch.setattr(llm_client, "chat_json", FauxLLM())
    monkeypatch.setattr(scoring, "encoder", faux_encoder)

    ingestion.ingerer("all")
    resume = storage.resume_offres().set_index("echelle")
    assert resume.loc["internationale", "nb_offres"] == 3
    assert resume.loc["nationale", "nb_offres"] == 4            # France + postes data uniquement
    o = storage.lire("SELECT * FROM offres WHERE source='data_jobs' AND titre='Stage Data Scientist'").iloc[0]
    assert (o["region"], o["type_contrat"], o["niveau_experience"]) == ("Auvergne-Rhône-Alpes", "Stage", "Stage")

    # Le CV est ajouté directement (pas de PDF dans le test)
    storage.enregistrer_profil({"profil_id": "aya", "texte_brut": "Data analyst : Python, SQL, Power BI, Excel."})

    assert extraction.extraire_offres() == 7
    assert extraction.extraire_cvs() == 1
    ext = storage.offres_avec_extraction("nationale")
    assert set(ext["modele"]) == {"source:data_jobs"}
    li1 = storage.offres_avec_extraction("internationale").set_index("offre_id").loc["li_1"]
    assert li1["competences"] == ["python", "sql"]            # normalisation en minuscules
    assert storage.profils().iloc[0]["competences"] == ["python", "sql", "power bi"]

    df = scoring.calculer(llm_top=1)
    assert len(df) == 7 and df["score_final"].between(0, 1).all()
    assert df["score_llm"].notna().sum() == 2                 # 1 offre jugée par échelle

    cl = aggregation.classement("aya", "nationale")
    assert cl.iloc[0]["titre"] == "Data Analyst F/H"          # sql + python + power bi : tout est acquis
    assert cl.iloc[0]["score_regles"] == 1.0

    comp = aggregation.comparaison_competences(5)
    assert {"python", "sql"} <= set(comp["competence"])
    g = aggregation.gap("aya", "nationale")
    assert not g.set_index("competence").loc["spark", "possedee"]
    assert 0 < aggregation.taux_couverture("aya", "nationale") < 1
    assert not aggregation.comparaison_methodes().empty
    assert set(aggregation.salaires("nationale")["devise"]) == {"USD"}


def test_bascule_data_jobs_vers_france_travail(env_complet, monkeypatch):
    """Passage de la source de secours à France Travail : purge, puis ingestion « all »."""
    from src.sources import france_travail as ft
    from tests.test_ingestion import FausseSession

    monkeypatch.setattr(llm_client, "chat_json", FauxLLM())
    monkeypatch.setattr(scoring, "encoder", faux_encoder)
    ingestion.ingerer("all")                                   # LinkedIn + data_jobs
    storage.enregistrer_profil({"profil_id": "aya", "texte_brut": "Python, SQL"})
    extraction.extraire_offres()
    scoring.calculer()

    session, vraie_collecte = FausseSession(), ft.collecter
    monkeypatch.setattr(ft, "collecter", lambda: vraie_collecte(mots_cles=["data scientist"], session=session))
    monkeypatch.setattr(ingestion, "SOURCE_NATIONALE", "france_travail")
    with pytest.raises(RuntimeError, match="purger data_jobs"):
        ingestion.ingerer("all")                               # refus : deux sources nationales

    assert storage.supprimer_source("data_jobs") == 4
    assert storage.lire("SELECT COUNT(*) n FROM matchings WHERE offre_id LIKE 'dj_%'")["n"][0] == 0
    ingestion.ingerer("all")
    assert set(storage.lire("SELECT DISTINCT source FROM offres WHERE echelle='nationale'")["source"]) \
        == {"france_travail"}
    assert len(storage.offres_non_extraites("nationale")) == 2  # 2 offres data FT, à passer au LLM


def test_normalisation():
    assert extraction.normaliser_liste(["MS Excel", "excel", " PowerBI ", "Python 3", "ML"]) == \
        ["excel", "power bi", "python", "machine learning"]
    assert scoring.present_dans_cv("r", "langages : python, r et sql")
    assert not scoring.present_dans_cv("r", "power bi")
    assert scoring.present_dans_cv("c++", "java, c++")


def test_json_depuis_texte():
    assert llm_client._json_depuis_texte('```json\n{"a": 1}\n```') == {"a": 1}
    assert llm_client._json_depuis_texte('Voici : {"a": [1, 2]} merci') == {"a": [1, 2]}
