"""Pénalité de niveau, journal d'extraction, métriques d'évaluation et comparaison de modèles."""
import pandas as pd
import pytest

from src import evaluation, extraction, ingestion, llm_client, scoring, storage
from tests.test_ingestion import env  # noqa: F401
from tests.test_pipeline import FauxLLM, env_complet, faux_encoder  # noqa: F401


def test_facteur_niveau_et_exclusions():
    assert scoring.facteur_niveau("Junior", "Junior")[0] == 1.0
    assert scoring.facteur_niveau("Stage", "Junior")[0] == 1.0
    assert scoring.facteur_niveau("Confirmé", "Junior")[0] == 0.75
    assert scoring.facteur_niveau("Senior", "Junior")[0] == 0.5
    assert scoring.facteur_niveau("Senior", "Stage")[0] == 0.5
    assert scoring.facteur_niveau(None, "Junior")[0] == 1.0
    assert scoring.facteur_exclusion("Data Scientist - TS/SCI Hybrid", "")[0] == 0.3
    assert scoring.facteur_exclusion("Volunteer: Data Engineer", "")[0] == 0.3
    assert scoring.facteur_exclusion("Data Analyst", "Security clearance required.")[0] == 0.3
    assert scoring.facteur_exclusion("Data Analyst (H/F)", "SQL et Python")[0] == 1.0


def test_metriques_unitaires():
    assert evaluation.precision_at_k([2, 2, 1, 0], k=2) == 1.0
    assert evaluation.precision_at_k([2, 1, 0, 2], k=4) == 0.5
    assert evaluation.precision_at_k([2, 1, 0, 2], k=4, seuil=1) == 0.75
    assert evaluation.ndcg_at_k([2, 1, 0]) == pytest.approx(1.0)
    assert evaluation.ndcg_at_k([0, 1, 2]) < 0.7
    assert evaluation.jaccard(["sql", "python"], ["sql"]) == 0.5


@pytest.fixture
def base_notee(env_complet, monkeypatch, tmp_path):  # noqa: F811
    monkeypatch.setattr(llm_client, "chat_json", FauxLLM())
    monkeypatch.setattr(scoring, "encoder", faux_encoder)
    monkeypatch.setattr(evaluation, "EVAL_DIR", tmp_path / "eval")
    monkeypatch.setattr(evaluation, "MODELS_DIR", tmp_path / "models")
    ingestion.ingerer("all")
    storage.enregistrer_profil({"profil_id": "aya", "texte_brut": "Data analyst : Python, SQL, Power BI, Excel."})
    extraction.extraire_offres()
    extraction.extraire_cvs()
    scoring.calculer()
    return tmp_path


def test_journal_et_penalite(base_notee):
    j = evaluation.stats_journal()
    assert j["appels"].sum() == 3 and (j["pct_json_valide_1er_essai"] == 100).all()
    # « Data Engineer Senior » (data_jobs) est pénalisé ×0,5 et l'ajustement est expliqué
    m = storage.lire("SELECT m.justification FROM matchings m JOIN offres o USING (offre_id) "
                     "WHERE o.titre = 'Data Engineer Senior'").iloc[0, 0]
    assert "×0.5" in m


def test_preparer_puis_calculer(base_notee):
    evaluation.preparer("aya", n_top=2, n_hasard=1)
    a = evaluation._lire_csv(evaluation.FICHIER_ANNOTATION)
    assert "score_final" not in a.columns and len(a) == 6        # 3 par échelle, sans les scores
    a["pertinence"] = [2, 0, 1, 2, 1, 0]
    a.to_csv(evaluation.EVAL_DIR / evaluation.FICHIER_ANNOTATION, **evaluation.CSV)
    c = evaluation._lire_csv(evaluation.FICHIER_CONTROLE)
    assert len(c) == 3                                            # seules les offres extraites par le LLM
    c["ok_competences"], c["ok_niveau"], c["ok_contrat"] = [1, 1, 0], [1, 0, 1], [1, 1, 1]
    c.to_csv(evaluation.EVAL_DIR / evaluation.FICHIER_CONTROLE, **evaluation.CSV)
    evaluation.preparer("aya")                                    # ne doit pas écraser les annotations
    assert evaluation._lire_csv(evaluation.FICHIER_ANNOTATION)["pertinence"].notna().all()

    evaluation.calculer()
    r = pd.read_csv(evaluation.EVAL_DIR / "resultats_matching.csv", sep=";", encoding="utf-8-sig")
    assert set(r["methode"]) >= {"Score final", "Règles", "Embeddings"}
    assert r["precision@10"].between(0, 1).all()
    x = pd.read_csv(evaluation.EVAL_DIR / "resultats_extraction.csv", sep=";", encoding="utf-8-sig")
    assert x["correct_contrat"].iloc[0] == 1.0


def test_comparer_modeles(base_notee):
    r = evaluation.comparer_modeles(["modele-a", "modele-b"], n=4)
    assert list(r["modele"]) == ["modele-a", "modele-b"]
    assert (r["accord_competences_jaccard"] == 1.0).all()        # faux LLM identique pour les deux
    assert (r["accord_niveau"] == 1.0).all() and (r["accord_contrat"] == 1.0).all()
    assert evaluation._egal(None, float("nan")) and not evaluation._egal(None, "CDI")
    assert (evaluation.MODELS_DIR / "comparaison_modeles.csv").exists()


def test_comparer_modeles_avec_echec(base_notee, monkeypatch):
    """Une offre ratée par la référence : pas de fausse alerte, et exclue des taux d'accord."""
    vrai, appels = llm_client.chat_json, {"n": 0}

    def llm_avec_echec(*a, **k):
        appels["n"] += 1
        if appels["n"] == 1:
            llm_client.dernier_appel.update(essais=3, succes=False, duree_s=0.1, erreur="x")
            raise RuntimeError("JSON invalide")
        r = vrai(*a, **k)
        llm_client.dernier_appel["modele_servi"] = k.get("modele")
        return r

    monkeypatch.setattr(llm_client, "chat_json", llm_avec_echec)
    r = evaluation.comparer_modeles(["modele-a", "modele-b"], n=4).set_index("modele")
    assert r.loc["modele-a", "modele_reellement_utilise"] == "modele-a"     # pas de « nan »
    assert r.loc["modele-b", "offres_comparees"] == r.loc["modele-b", "offres"] - 1
    assert r.loc["modele-b", "accord_competences_jaccard"] == 1.0
