"""Échelle nationale : API Offres d'emploi v2 de France Travail.

Accès : créer un compte sur https://francetravail.io, déclarer une application, l'abonner à
l'API « Offres d'emploi v2 », puis mettre FT_CLIENT_ID et FT_CLIENT_SECRET dans le fichier .env.

Contraintes de l'API gérées ici :
- authentification OAuth2 (client credentials), jeton de courte durée ;
- 150 offres par page, 1 150 offres au plus par requête -> une requête par mot-clé ;
- environ 10 appels par seconde -> petite pause entre les pages.

Les réponses brutes sont sauvegardées dans data/raw/france_travail/ : on peut retraiter
sans rappeler l'API, et la date de collecte reste tracée (obligation de citer la source).
"""
import json
import re
import time
from datetime import date

import pandas as pd
import requests

from src.config import (
    FT_CLIENT_ID, FT_CLIENT_SECRET, FT_MAX_INDEX, FT_MOTS_CLES, FT_PAGE, FT_PAUSE_S,
    FT_RAW_DIR, FT_SCOPE, FT_SEARCH_URL, FT_TOKEN_URL, SEED, TAILLE_ECHANTILLON,
    TITRES_CIBLES_FR,
)

# --- Référentiels -------------------------------------------------------------------------

REGIONS = {
    "Auvergne-Rhône-Alpes": "01 03 07 15 26 38 42 43 63 69 73 74",
    "Bourgogne-Franche-Comté": "21 25 39 58 70 71 89 90",
    "Bretagne": "22 29 35 56",
    "Centre-Val de Loire": "18 28 36 37 41 45",
    "Corse": "2A 2B",
    "Grand Est": "08 10 51 52 54 55 57 67 68 88",
    "Hauts-de-France": "02 59 60 62 80",
    "Île-de-France": "75 77 78 91 92 93 94 95",
    "Normandie": "14 27 50 61 76",
    "Nouvelle-Aquitaine": "16 17 19 23 24 33 40 47 64 79 86 87",
    "Occitanie": "09 11 12 30 31 32 34 46 48 65 66 81 82",
    "Pays de la Loire": "44 49 53 72 85",
    "Provence-Alpes-Côte d'Azur": "04 05 06 13 83 84",
    "Guadeloupe": "971", "Martinique": "972", "Guyane": "973", "La Réunion": "974", "Mayotte": "976",
}
DEPT_VERS_REGION = {d: r for r, depts in REGIONS.items() for d in depts.split()}

CONTRATS = {"CDI": "CDI", "CDD": "CDD", "MIS": "Intérim", "SAI": "CDD", "LIB": "Freelance"}

# Heures annuelles légales d'un temps plein en France (35 h).
HEURES_AN = 1607


# --- Parsing des champs -------------------------------------------------------------------

def region_depuis_lieu(libelle: str | None) -> tuple[str | None, str | None]:
    """'75 - Paris 15e Arrondissement' -> ('Île-de-France', 'Paris 15e Arrondissement')."""
    if not libelle:
        return None, None
    m = re.match(r"^\s*(\d{3}|\d{2}|2[AB])\s*-\s*(.*)$", libelle)
    if m:
        return DEPT_VERS_REGION.get(m.group(1)), m.group(2).strip() or None
    # Offres rattachées directement à une région ('Ile-de-France') ou à 'France'
    for region in REGIONS:
        if libelle.strip().lower().replace("î", "i") == region.lower().replace("î", "i"):
            return region, None
    return None, None


def salaire_annuel(libelle: str | None) -> float | None:
    """Annualise le libellé de salaire France Travail (milieu de fourchette, en euros bruts).

    'Annuel de 35000.00 Euros à 45000.00 Euros sur 12 mois' -> 40000
    'Mensuel de 2500.00 Euros sur 13 mois'                  -> 32500
    'Horaire de 12.61 Euros à 13.87 Euros'                  -> 21276.86
    """
    if not libelle:
        return None
    texte = libelle.replace(",", ".")
    m = re.match(r"\s*(Annuel|Mensuel|Horaire)\s+de\s+([\d.]+)\s*Euros(?:\s+à\s+([\d.]+)\s*Euros)?", texte, re.I)
    if not m:
        return None
    periode, bas, haut = m.group(1).lower(), float(m.group(2)), m.group(3)
    montant = (bas + float(haut)) / 2 if haut else bas
    if periode == "annuel":
        return round(montant, 2)
    if periode == "mensuel":
        mois = re.search(r"sur\s+([\d.]+)\s*mois", texte)
        return round(montant * (float(mois.group(1)) if mois else 12), 2)
    return round(montant * HEURES_AN, 2)


def niveau_experience(offre: dict) -> str | None:
    if offre.get("experienceExige") == "D":
        return "Débutant"
    texte = offre.get("experienceLibelle") or ""
    m = re.search(r"(\d+)\s*An", texte)
    if m:
        annees = int(m.group(1))
    else:
        m = re.search(r"(\d+)\s*Mois", texte)
        annees = int(m.group(1)) / 12 if m else None
    if annees is None:
        return None
    if annees <= 2:
        return "Junior"
    return "Confirmé" if annees <= 5 else "Senior"


def type_contrat(offre: dict) -> str:
    if offre.get("alternance"):
        return "Alternance"
    return CONTRATS.get(offre.get("typeContrat"), "Autre")


def vers_schema_commun(offres: list[dict]) -> pd.DataFrame:
    lignes = []
    for o in offres:
        lieu = o.get("lieuTravail") or {}
        region, ville = region_depuis_lieu(lieu.get("libelle"))
        url = (o.get("origineOffre") or {}).get("urlOrigine") or (
            f"https://candidat.francetravail.fr/offres/recherche/detail/{o['id']}"
        )
        lignes.append({
            "offre_id": f"ft_{o['id']}",
            "source": "france_travail",
            "echelle": "nationale",
            "langue": "fr",
            "titre": o.get("intitule"),
            "description": o.get("description"),
            "entreprise": (o.get("entreprise") or {}).get("nom"),
            "pays": "France",
            "region": region,
            "ville": ville,
            "type_contrat": type_contrat(o),
            "niveau_experience": niveau_experience(o),
            "teletravail": None,   # non publié de façon fiable par l'API
            "salaire_annuel": salaire_annuel((o.get("salaire") or {}).get("libelle")),
            "devise": "EUR",
            "date_publication": (o.get("dateCreation") or "")[:10] or None,
            "url": url,
            "code_rome": o.get("romeCode"),
            "competences_source": [c.get("libelle") for c in o.get("competences") or [] if c.get("libelle")] or None,
        })
    return pd.DataFrame(lignes)


# --- Appels à l'API -----------------------------------------------------------------------

def obtenir_jeton(session: requests.Session) -> str:
    if not (FT_CLIENT_ID and FT_CLIENT_SECRET):
        raise RuntimeError(
            "FT_CLIENT_ID / FT_CLIENT_SECRET absents : crée une application sur francetravail.io "
            "et renseigne-les dans le fichier .env (voir .env.example)."
        )
    r = session.post(
        FT_TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": FT_CLIENT_ID,
            "client_secret": FT_CLIENT_SECRET,
            "scope": FT_SCOPE,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["access_token"]


def rechercher(session: requests.Session, jeton: str, mots_cles: str, **filtres) -> list[dict]:
    """Toutes les pages d'une requête (1 150 offres au plus). 200/206 = résultats, 204 = aucun."""
    offres, debut = [], 0
    while debut <= FT_MAX_INDEX:
        fin = min(debut + FT_PAGE - 1, FT_MAX_INDEX)
        r = session.get(
            FT_SEARCH_URL,
            headers={"Authorization": f"Bearer {jeton}", "Accept": "application/json"},
            params={"motsCles": mots_cles, "range": f"{debut}-{fin}", **filtres},
            timeout=30,
        )
        if r.status_code == 204:
            break
        r.raise_for_status()
        page = r.json().get("resultats", [])
        offres.extend(page)
        total = _total(r.headers.get("Content-Range"))
        if len(page) < FT_PAGE or (total is not None and fin + 1 >= total):
            break
        debut += FT_PAGE
        time.sleep(FT_PAUSE_S)
    return offres


def _total(content_range: str | None) -> int | None:
    m = re.search(r"/(\d+)", content_range or "")
    return int(m.group(1)) if m else None


def collecter(mots_cles=FT_MOTS_CLES, session: requests.Session | None = None, **filtres) -> list[dict]:
    """Lance une requête par mot-clé, dédoublonne, sauvegarde le brut en JSON."""
    session = session or requests.Session()
    jeton = obtenir_jeton(session)
    par_id = {}
    for mc in mots_cles:
        res = rechercher(session, jeton, mc, **filtres)
        nouvelles = sum(1 for o in res if o["id"] not in par_id)
        par_id.update({o["id"]: o for o in res})
        print(f"[France Travail] '{mc}' : {len(res)} offres ({nouvelles} nouvelles)")
        time.sleep(FT_PAUSE_S)

    FT_RAW_DIR.mkdir(parents=True, exist_ok=True)
    fichier = FT_RAW_DIR / f"offres_{date.today().isoformat()}.json"
    fichier.write_text(json.dumps(list(par_id.values()), ensure_ascii=False), encoding="utf-8")
    print(f"[France Travail] {len(par_id)} offres uniques sauvegardées dans {fichier.name}")
    return list(par_id.values())


def charger_brut() -> list[dict]:
    """Relit toutes les collectes sauvegardées (la plus récente l'emporte en cas de doublon)."""
    par_id = {}
    for f in sorted(FT_RAW_DIR.glob("offres_*.json")):
        par_id.update({o["id"]: o for o in json.loads(f.read_text(encoding="utf-8"))})
    return list(par_id.values())


def charger(appeler_api: bool = True, taille: int = TAILLE_ECHANTILLON) -> pd.DataFrame:
    offres = collecter() if appeler_api else charger_brut()
    if not offres:
        raise RuntimeError("Aucune offre France Travail : lance d'abord une collecte.")
    df = vers_schema_commun(offres)
    total = len(df)
    df = df[df["titre"].str.contains(TITRES_CIBLES_FR, case=False, na=False, regex=True)]
    df = df.dropna(subset=["description"])
    retenues = len(df)
    if len(df) > taille:
        df = df.sample(n=taille, random_state=SEED)
    print(f"[France Travail] {total} offres, {retenues} postes data ciblés, {len(df)} gardées")
    return df.reset_index(drop=True)
