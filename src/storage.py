"""Stockage SQLite : offres, profils, extractions, matchings et suivi des candidatures."""
import json
import sqlite3
from contextlib import contextmanager

import pandas as pd

from src.config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS offres (
    job_id              INTEGER PRIMARY KEY,
    company_name        TEXT,
    title               TEXT,
    description         TEXT,
    location            TEXT,
    etat                TEXT,
    work_type           TEXT,
    experience_level    TEXT,
    remote_allowed      INTEGER,
    salaire_annuel      REAL,
    currency            TEXT,
    date_publication    TEXT,
    url                 TEXT,
    categories_linkedin TEXT
);

CREATE TABLE IF NOT EXISTS offres_extraites (
    job_id              INTEGER PRIMARY KEY REFERENCES offres(job_id),
    competences_requises TEXT,   -- JSON (liste)
    niveau_exp          TEXT,
    type_contrat        TEXT,
    localisation        TEXT,
    salaire_estime      REAL,
    resume              TEXT,
    modele              TEXT,
    date_extraction     TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS profils (
    profil_id           TEXT PRIMARY KEY,   -- nom du fichier CV sans extension
    texte_brut          TEXT,
    competences         TEXT,   -- JSON
    experiences         TEXT,   -- JSON
    formation           TEXT,   -- JSON
    modele              TEXT,
    date_extraction     TEXT
);

CREATE TABLE IF NOT EXISTS matchings (
    profil_id           TEXT REFERENCES profils(profil_id),
    job_id              INTEGER REFERENCES offres(job_id),
    score_embedding     REAL,
    score_regles        REAL,
    score_llm           REAL,
    score_final         REAL,
    justification       TEXT,
    date_calcul         TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (profil_id, job_id)
);

CREATE TABLE IF NOT EXISTS suivi (
    profil_id           TEXT REFERENCES profils(profil_id),
    job_id              INTEGER REFERENCES offres(job_id),
    statut              TEXT CHECK (statut IN ('vue', 'a_postuler', 'postulee', 'refusee', 'entretien')),
    date_maj            TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (profil_id, job_id)
);
"""


@contextmanager
def connexion():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with connexion() as conn:
        conn.executescript(SCHEMA)


def _upsert(table, lignes, cle):
    """Insère ou met à jour une liste de dicts dans une table."""
    if not lignes:
        return
    colonnes = list(lignes[0].keys())
    maj = ", ".join(f"{c}=excluded.{c}" for c in colonnes if c not in cle)
    sql = (
        f"INSERT INTO {table} ({', '.join(colonnes)}) "
        f"VALUES ({', '.join('?' * len(colonnes))}) "
        f"ON CONFLICT({', '.join(cle)}) DO UPDATE SET {maj}"
    )
    with connexion() as conn:
        conn.executemany(sql, [tuple(l[c] for c in colonnes) for l in lignes])


def _json(valeur):
    return json.dumps(valeur, ensure_ascii=False) if isinstance(valeur, (list, dict)) else valeur


# --- Écriture ---
def enregistrer_offres(df: pd.DataFrame):
    lignes = df.astype(object).where(df.notna(), None).to_dict("records")
    _upsert("offres", lignes, ["job_id"])


def enregistrer_profils(df: pd.DataFrame):
    lignes = df.astype(object).where(df.notna(), None).to_dict("records")
    _upsert("profils", lignes, ["profil_id"])


def enregistrer_extraction_offre(job_id: int, data: dict, modele: str):
    ligne = {"job_id": job_id, "modele": modele, **{k: _json(v) for k, v in data.items()}}
    _upsert("offres_extraites", [ligne], ["job_id"])


def enregistrer_matchings(lignes: list[dict]):
    _upsert("matchings", lignes, ["profil_id", "job_id"])


def maj_statut(profil_id: str, job_id: int, statut: str):
    _upsert("suivi", [{"profil_id": profil_id, "job_id": job_id, "statut": statut}], ["profil_id", "job_id"])


# --- Lecture ---
def lire(requete: str, params: tuple = ()) -> pd.DataFrame:
    with connexion() as conn:
        return pd.read_sql_query(requete, conn, params=params)


def offres_non_extraites(limite: int | None = None) -> pd.DataFrame:
    sql = """
        SELECT o.job_id, o.title, o.description
        FROM offres o LEFT JOIN offres_extraites e USING (job_id)
        WHERE e.job_id IS NULL
    """
    if limite:
        sql += f" LIMIT {int(limite)}"
    return lire(sql)
