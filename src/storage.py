"""Stockage SQLite : offres (deux échelles), extractions, profils, matchings et suivi.

Chaque offre a un identifiant texte préfixé par sa source :
    li_<job_id LinkedIn>      -> échelle internationale
    ft_<id France Travail>    -> échelle nationale
"""
import json
import sqlite3
from contextlib import contextmanager

import pandas as pd

from src.config import DB_PATH

ECHELLES = ("internationale", "nationale")

# Colonnes du schéma commun, dans l'ordre. Les deux sources produisent exactement ces colonnes.
COLONNES_OFFRES = [
    "offre_id", "source", "echelle", "langue",
    "titre", "description", "entreprise",
    "pays", "region", "ville",
    "type_contrat", "niveau_experience", "teletravail",
    "salaire_annuel", "devise",
    "date_publication", "url",
    "code_rome", "competences_source",
]

SCHEMA = """
CREATE TABLE IF NOT EXISTS offres (
    offre_id            TEXT PRIMARY KEY,
    source              TEXT NOT NULL,          -- linkedin | france_travail
    echelle             TEXT NOT NULL CHECK (echelle IN ('internationale', 'nationale')),
    langue              TEXT,                   -- en | fr
    titre               TEXT,
    description         TEXT,
    entreprise          TEXT,
    pays                TEXT,
    region              TEXT,                   -- État américain ou région française
    ville               TEXT,
    type_contrat        TEXT,                   -- CDI, CDD, Stage, Alternance, Freelance, Intérim, Autre
    niveau_experience   TEXT,                   -- Stage, Débutant, Junior, Confirmé, Senior
    teletravail         INTEGER,
    salaire_annuel      REAL,                   -- dans la devise de l'offre
    devise              TEXT,
    date_publication    TEXT,
    url                 TEXT,
    code_rome           TEXT,                   -- France Travail uniquement
    competences_source  TEXT                    -- JSON : compétences fournies par la source
);
CREATE INDEX IF NOT EXISTS idx_offres_echelle ON offres(echelle);

CREATE TABLE IF NOT EXISTS offres_extraites (
    offre_id             TEXT PRIMARY KEY REFERENCES offres(offre_id),
    competences_requises TEXT,   -- JSON (liste normalisée)
    niveau_exp           TEXT,
    type_contrat         TEXT,
    localisation         TEXT,
    salaire_estime       REAL,
    resume               TEXT,
    modele               TEXT,
    date_extraction      TEXT DEFAULT CURRENT_TIMESTAMP
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
    offre_id            TEXT REFERENCES offres(offre_id),
    score_embedding     REAL,
    score_regles        REAL,
    score_llm           REAL,
    score_final         REAL,
    justification       TEXT,
    date_calcul         TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (profil_id, offre_id)
);

CREATE TABLE IF NOT EXISTS suivi (
    profil_id           TEXT REFERENCES profils(profil_id),
    offre_id            TEXT REFERENCES offres(offre_id),
    statut              TEXT CHECK (statut IN ('vue', 'a_postuler', 'postulee', 'refusee', 'entretien')),
    date_maj            TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (profil_id, offre_id)
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
        # L'ancien schéma (job_id entier, LinkedIn seul) n'est pas compatible : on le signale.
        cols = [r[1] for r in conn.execute("PRAGMA table_info(offres)")]
        if cols and "offre_id" not in cols:
            raise RuntimeError(
                f"{DB_PATH} utilise l'ancien schéma. Supprime ce fichier puis relance l'ingestion."
            )
        conn.executescript(SCHEMA)


def _upsert(table, lignes, cle):
    """Insère ou met à jour une liste de dicts : relancer ne crée pas de doublons."""
    if not lignes:
        return 0
    colonnes = list(lignes[0].keys())
    maj = ", ".join(f"{c}=excluded.{c}" for c in colonnes if c not in cle)
    sql = (
        f"INSERT INTO {table} ({', '.join(colonnes)}) "
        f"VALUES ({', '.join('?' for _ in colonnes)}) "
        f"ON CONFLICT ({', '.join(cle)}) DO "
        + (f"UPDATE SET {maj}" if maj else "NOTHING")
    )
    with connexion() as conn:
        conn.executemany(sql, [tuple(l[c] for c in colonnes) for l in lignes])
    return len(lignes)


def _propre(v):
    """Convertit NaN / numpy en types acceptés par SQLite."""
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    if hasattr(v, "item"):
        return v.item()
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)
    return v


def enregistrer_offres(df: pd.DataFrame) -> int:
    manquantes = set(COLONNES_OFFRES) - set(df.columns)
    if manquantes:
        raise ValueError(f"Colonnes manquantes pour le schéma commun : {sorted(manquantes)}")
    lignes = [
        {c: _propre(r[c]) for c in COLONNES_OFFRES}
        for r in df[COLONNES_OFFRES].to_dict("records")
    ]
    return _upsert("offres", lignes, ["offre_id"])


def enregistrer_profil(profil: dict) -> int:
    return _upsert("profils", [{k: _propre(v) for k, v in profil.items()}], ["profil_id"])


def lire(sql: str, params=()) -> pd.DataFrame:
    with connexion() as conn:
        return pd.read_sql_query(sql, conn, params=params)


def offres_non_extraites(echelle: str | None = None, limite: int | None = None) -> pd.DataFrame:
    """Offres pas encore passées dans le LLM (étape 2), éventuellement pour une seule échelle."""
    sql = (
        "SELECT o.* FROM offres o LEFT JOIN offres_extraites e USING (offre_id) "
        "WHERE e.offre_id IS NULL"
    )
    params = []
    if echelle:
        sql += " AND o.echelle = ?"
        params.append(echelle)
    sql += " ORDER BY o.offre_id"
    if limite:
        sql += f" LIMIT {int(limite)}"
    return lire(sql, params)


def resume_offres() -> pd.DataFrame:
    """Vue d'ensemble par échelle : volume, taux de salaire renseigné, contrats."""
    return lire(
        """
        SELECT echelle, source, langue,
               COUNT(*)                                   AS nb_offres,
               ROUND(AVG(salaire_annuel IS NOT NULL), 3)  AS taux_salaire,
               COUNT(DISTINCT region)                     AS nb_regions,
               MIN(date_publication)                      AS premiere_date,
               MAX(date_publication)                      AS derniere_date
        FROM offres GROUP BY echelle, source, langue ORDER BY echelle
        """
    )
