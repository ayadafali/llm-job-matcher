"""Tests de la couche données sur des données simulées (aucun appel réseau).

Lancer : python -m pytest -q
"""
import json

import pandas as pd
import pytest

from src import ingestion, storage
from src.sources import france_travail as ft
from src.sources import linkedin as li


# --- Fixtures ------------------------------------------------------------------------------

@pytest.fixture
def env(tmp_path, monkeypatch):
    raw, processed = tmp_path / "raw", tmp_path / "processed"
    (raw / "jobs").mkdir(parents=True)
    (raw / "mappings").mkdir()
    (raw / "cv").mkdir()
    processed.mkdir()
    monkeypatch.setattr(storage, "DB_PATH", processed / "test.db")
    monkeypatch.setattr(li, "RAW_DIR", raw)
    monkeypatch.setattr(ft, "FT_RAW_DIR", raw / "france_travail")
    monkeypatch.setattr(ft, "FT_CLIENT_ID", "id-test")
    monkeypatch.setattr(ft, "FT_CLIENT_SECRET", "secret-test")
    monkeypatch.setattr(ft, "FT_PAUSE_S", 0)
    monkeypatch.setattr(ingestion, "PROCESSED_DIR", processed)
    monkeypatch.setattr(ingestion, "CV_DIR", raw / "cv")

    pd.DataFrame({
        "job_id": [1, 2, 3, 4],
        "company_name": ["A", "B", "C", "D"],
        "title": ["Data Analyst", "Sales Rep", "Junior Data Scientist", "BI Analyst Intern"],
        "description": ["SQL, Python", "sell", "ML, Python", "Power BI"],
        "location": ["Austin, TX", "New York, NY", "United States", "Boston, MA"],
        "formatted_work_type": ["Full-time", "Full-time", "Contract", "Internship"],
        "formatted_experience_level": ["Entry level", "Entry level", "Associate", "Internship"],
        "remote_allowed": [1, None, None, None],
        "min_salary": [50000, None, None, None],
        "max_salary": [70000, None, None, None],
        "med_salary": [None, None, None, 25],
        "pay_period": ["YEARLY", None, None, "HOURLY"],
        "normalized_salary": [None, None, None, None],
        "currency": ["USD"] * 4,
        "listed_time": [1713398000000] * 4,
        "job_posting_url": ["u1", "u2", "u3", "u4"],
    }).to_csv(raw / "postings.csv", index=False)
    pd.DataFrame({"job_id": [1, 1, 3], "skill_abr": ["IT", "ANLS", "IT"]}).to_csv(
        raw / "jobs" / "job_skills.csv", index=False)
    pd.DataFrame({"skill_abr": ["IT", "ANLS"], "skill_name": ["Information Technology", "Analyst"]}).to_csv(
        raw / "mappings" / "skills.csv", index=False)
    return tmp_path


def offre_ft(i, titre="Data analyst (H/F)", lieu="75 - Paris 15e Arrondissement", **extra):
    o = {
        "id": f"{200 + i}ABC",
        "intitule": titre,
        "description": "Analyse de données avec SQL et Python.",
        "dateCreation": "2026-11-10T09:00:00.000Z",
        "lieuTravail": {"libelle": lieu},
        "entreprise": {"nom": f"Entreprise {i}"},
        "typeContrat": "CDI",
        "experienceExige": "D",
        "experienceLibelle": "Débutant accepté",
        "romeCode": "M1403",
        "competences": [{"code": "1", "libelle": "Analyser des données"}],
        "salaire": {"libelle": "Annuel de 35000.00 Euros à 45000.00 Euros sur 12 mois"},
    }
    o.update(extra)
    return o


class FausseReponse:
    def __init__(self, status=200, data=None, headers=None):
        self.status_code, self._data, self.headers = status, data, headers or {}

    def json(self):
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class FausseSession:
    """Simule l'API : 'data analyst' renvoie 160 offres sur 2 pages, 'data scientist' 3 offres
    dont 1 doublon, les autres mots-clés aucune (204)."""

    def __init__(self):
        self.appels = []
        analyst = [offre_ft(i) for i in range(160)]
        scientist = [offre_ft(0), offre_ft(500, "Data scientist junior H/F", "69 - Lyon 3e"),
                     offre_ft(501, "Commercial terrain", "13 - Marseille")]
        self.donnees = {"data analyst": analyst, "data scientist": scientist}

    def post(self, url, data=None, headers=None, timeout=None):
        assert data["grant_type"] == "client_credentials"
        return FausseReponse(data={"access_token": "jeton", "expires_in": 1499})

    def get(self, url, headers=None, params=None, timeout=None):
        assert headers["Authorization"] == "Bearer jeton"
        self.appels.append(params)
        offres = self.donnees.get(params["motsCles"], [])
        if not offres:
            return FausseReponse(204)
        debut, fin = map(int, params["range"].split("-"))
        page = offres[debut:fin + 1]
        return FausseReponse(206 if len(offres) > fin + 1 else 200, {"resultats": page},
                             {"Content-Range": f"offres {debut}-{debut + len(page) - 1}/{len(offres)}"})


# --- Parsing France Travail ----------------------------------------------------------------

@pytest.mark.parametrize("libelle,attendu", [
    ("Annuel de 35000.00 Euros à 45000.00 Euros sur 12 mois", 40000),
    ("Mensuel de 2500.00 Euros sur 13 mois", 32500),
    ("Mensuel de 2000.00 Euros à 2400.00 Euros", 26400),
    ("Horaire de 12.61 Euros à 13.87 Euros", round(13.24 * 1607, 2)),
    ("Selon profil", None),
    (None, None),
])
def test_salaire_ft(libelle, attendu):
    assert ft.salaire_annuel(libelle) == attendu


@pytest.mark.parametrize("libelle,attendu", [
    ("75 - Paris 15e Arrondissement", ("Île-de-France", "Paris 15e Arrondissement")),
    ("2A - Ajaccio", ("Corse", "Ajaccio")),
    ("974 - Saint-Denis", ("La Réunion", "Saint-Denis")),
    ("69 - Rhône", ("Auvergne-Rhône-Alpes", "Rhône")),
    ("Ile-de-France", ("Île-de-France", None)),
    ("France", (None, None)),
])
def test_region_ft(libelle, attendu):
    assert ft.region_depuis_lieu(libelle) == attendu


def test_departements_complets():
    assert len(ft.DEPT_VERS_REGION) == 101   # 96 en métropole + 5 en outre-mer


def test_niveau_et_contrat_ft():
    assert ft.niveau_experience({"experienceExige": "E", "experienceLibelle": "3 An(s)"}) == "Confirmé"
    assert ft.niveau_experience({"experienceExige": "S", "experienceLibelle": "6 Mois"}) == "Junior"
    assert ft.type_contrat({"typeContrat": "CDD", "alternance": True}) == "Alternance"
    assert ft.type_contrat({"typeContrat": "MIS"}) == "Intérim"


# --- Collecte France Travail (API simulée) -------------------------------------------------

def test_collecte_pagination_et_doublons(env):
    session = FausseSession()
    offres = ft.collecter(mots_cles=["data analyst", "data scientist", "big data"], session=session)
    # 160 + 3 - 1 doublon
    assert len(offres) == 162
    ranges = [a["range"] for a in session.appels if a["motsCles"] == "data analyst"]
    assert ranges == ["0-149", "150-299"]
    assert len(list((env / "raw" / "france_travail").glob("offres_*.json"))) == 1


# --- Bout en bout ------------------------------------------------------------------------------

def test_bout_en_bout(env, monkeypatch):
    session, vraie_collecte = FausseSession(), ft.collecter
    monkeypatch.setattr(ft, "collecter", lambda: vraie_collecte(
        mots_cles=["data analyst", "data scientist"], session=session))
    ingestion.ingerer("all")

    resume = storage.resume_offres().set_index("echelle")
    assert resume.loc["internationale", "nb_offres"] == 3      # Sales Rep exclu
    assert resume.loc["nationale", "nb_offres"] == 161         # 'Commercial terrain' exclu

    o = storage.lire("SELECT * FROM offres WHERE offre_id = 'li_4'").iloc[0]
    assert o["salaire_annuel"] == 25 * 2080 and o["niveau_experience"] == "Stage"
    o = storage.lire("SELECT * FROM offres WHERE offre_id = 'li_1'").iloc[0]
    assert o["region"] == "TX" and json.loads(o["competences_source"]) == ["Analyst", "Information Technology"]
    o = storage.lire("SELECT * FROM offres WHERE offre_id = 'ft_700ABC'").iloc[0]
    assert (o["region"], o["devise"], o["langue"]) == ("Auvergne-Rhône-Alpes", "EUR", "fr")

    # Relancer ne duplique rien, et le mode hors ligne relit le JSON sauvegardé
    ingestion.ingerer("france_travail", hors_ligne=True)
    assert storage.lire("SELECT COUNT(*) n FROM offres")["n"][0] == 164
    assert len(storage.offres_non_extraites("nationale")) == 161


def test_ancien_schema_detecte(env):
    import sqlite3
    conn = sqlite3.connect(storage.DB_PATH)
    conn.execute("CREATE TABLE offres (job_id INTEGER PRIMARY KEY, title TEXT)")
    conn.commit(); conn.close()
    with pytest.raises(RuntimeError, match="ancien schéma"):
        storage.init_db()
