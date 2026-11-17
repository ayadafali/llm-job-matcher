"""Étape 3 — Score de matching CV / offre, par trois méthodes comparées.

1. Embeddings  : similarité cosinus entre le CV et l'offre (modèle multilingue multilingual-e5-base,
                 déjà entraîné : il gère les offres anglaises ET françaises face à un CV français).
2. Règles      : part des compétences demandées par l'offre que le CV possède.
3. LLM direct  : (optionnel, sur les meilleures offres seulement) le LLM local note l'adéquation
                 de 0 à 100 et justifie.

Score final = (0,6 x règles + 0,4 x embeddings remis à l'échelle 0-1 pour chaque profil)
              x pénalité de niveau (offre Confirmé/Senior pour un profil junior)
              x pénalité d'incompatibilité (habilitation de sécurité, bénévolat).

Exemples :
    python -m src.scoring                    # tous les profils, toutes les offres extraites
    python -m src.scoring --llm-top 10       # + jugement du LLM sur les 10 meilleures offres par échelle
"""
import argparse
import re

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src import llm_client, storage
from src.config import EMBEDDING_MODEL, PROFIL_NIVEAU

POIDS_REGLES, POIDS_EMBEDDING = 0.6, 0.4

_modele = None


def encoder(textes: list[str], role: str) -> np.ndarray:
    """Vecteurs normalisés. e5 attend le préfixe 'query: ' (le CV) ou 'passage: ' (les offres)."""
    global _modele
    if _modele is None:
        from sentence_transformers import SentenceTransformer
        _modele = SentenceTransformer(EMBEDDING_MODEL)
    return _modele.encode([f"{role}: {t}" for t in textes], normalize_embeddings=True,
                          batch_size=32, show_progress_bar=len(textes) > 64)


def texte_offre(o: dict) -> str:
    return f"{o['titre']}. {o.get('resume') or ''} Compétences : {', '.join(o['competences'])}."[:2000]


def texte_profil(p: dict) -> str:
    return f"{p.get('texte_brut') or ''}"[:3000] + " Compétences : " + ", ".join(p["competences"])


def competences_du_profil(p: dict) -> set[str]:
    """Compétences extraites du CV + toute compétence connue citée mot pour mot dans le CV."""
    return set(p["competences"])


def present_dans_cv(competence: str, texte_cv: str) -> bool:
    return re.search(rf"(?<![\w+#]){re.escape(competence)}(?![\w+#])", texte_cv) is not None


def score_regles(comp_offre: list[str], comp_cv: set[str], texte_cv: str) -> tuple[float | None, list[str], list[str]]:
    if not comp_offre:
        return None, [], []
    acquises = [c for c in comp_offre if c in comp_cv or present_dans_cv(c, texte_cv)]
    manquantes = [c for c in comp_offre if c not in acquises]
    return len(acquises) / len(comp_offre), acquises, manquantes


# --- Règles métier : niveau d'expérience et offres incompatibles ------------------------------

ORDRE_NIVEAUX = {"Stage": 0, "Débutant": 1, "Junior": 2, "Confirmé": 3, "Senior": 4}
# Pénalité selon le nombre de niveaux qui séparent l'offre (au-dessus) du profil
PENALITE_ECART = {1: 0.75, 2: 0.5}
PENALITE_ECART_MAX = 0.5

# Offres qu'un profil étudiant ne peut pas viser, quelles que soient ses compétences
EXCLUSIONS = [
    (r"clearance|ts/sci|top secret|secret clearance|habilitation (?:secret|défense)", "habilitation de sécurité exigée"),
    (r"\bvolunteer\b|\bbénévol", "poste bénévole"),
]
PENALITE_EXCLUSION = 0.3


def facteur_niveau(niveau_offre: str | None, niveau_profil: str = PROFIL_NIVEAU) -> tuple[float, str]:
    """1 si l'offre est au niveau du profil ou en dessous, pénalité sinon. Niveau inconnu : pas de pénalité."""
    if niveau_offre not in ORDRE_NIVEAUX:
        return 1.0, ""
    ecart = ORDRE_NIVEAUX[niveau_offre] - ORDRE_NIVEAUX[niveau_profil]
    if ecart <= 0:
        return 1.0, ""
    f = PENALITE_ECART.get(ecart, PENALITE_ECART_MAX)
    return f, f"niveau {niveau_offre} pour un profil {niveau_profil} (×{f:g})."


def facteur_exclusion(titre: str | None, description: str | None) -> tuple[float, str]:
    texte = f"{titre or ''} {(description or '')[:3000]}".lower()
    for motif, raison in EXCLUSIONS:
        if re.search(motif, texte):
            return PENALITE_EXCLUSION, f"{raison} (×{PENALITE_EXCLUSION:g})."
    return 1.0, ""


class JugementLLM(BaseModel):
    score: int = Field(description="Adéquation du profil à l'offre, de 0 à 100")
    justification: str = Field(description="Deux phrases au plus, en français")


PROMPT_JUGEMENT = """Tu es recruteur. Évalue l'adéquation entre le profil d'un candidat et une offre d'emploi.
Note de 0 (aucun rapport) à 100 (profil idéal). Tiens compte des compétences, du niveau
d'expérience et du type de poste. Réponds en JSON, justification en français (deux phrases au plus)."""


def juger(profil: dict, offre: dict) -> JugementLLM:
    texte = (f"PROFIL :\n{texte_profil(profil)[:2500]}\n\n"
             f"OFFRE : {offre['titre']} ({offre.get('entreprise') or ''})\n"
             f"Niveau : {offre.get('niveau_exp') or 'non précisé'} ; contrat : {offre.get('contrat_extrait') or 'non précisé'}\n"
             f"{offre.get('resume') or ''}\nCompétences demandées : {', '.join(offre['competences'])}")
    return llm_client.chat_json(PROMPT_JUGEMENT, texte, JugementLLM)


def calculer(llm_top: int = 0) -> pd.DataFrame:
    offres = storage.offres_avec_extraction()
    profs = storage.profils()
    if offres.empty or profs.empty:
        raise RuntimeError("Il faut des offres extraites (étape 2) et au moins un CV ingéré.")

    vecteurs_offres = encoder([texte_offre(o) for o in offres.to_dict("records")], "passage")
    resultats = []
    for p in profs.to_dict("records"):
        texte_cv = (p.get("texte_brut") or "").lower()
        comp_cv = competences_du_profil(p)
        v_cv = encoder([texte_profil(p)], "query")[0]
        cos = vecteurs_offres @ v_cv
        cos_01 = (cos - cos.min()) / (cos.max() - cos.min() + 1e-9)

        lignes = []
        for o, c, c01 in zip(offres.to_dict("records"), cos, cos_01):
            regles, acquises, manquantes = score_regles(o["competences"], comp_cv, texte_cv)
            base = c01 if regles is None else POIDS_REGLES * regles + POIDS_EMBEDDING * c01
            f_niveau, motif_niveau = facteur_niveau(o.get("niveau_exp") or o.get("niveau_experience"))
            f_exclu, motif_exclu = facteur_exclusion(o["titre"], o.get("description"))
            final = base * f_niveau * f_exclu
            ajustements = " ".join(m for m in (motif_niveau, motif_exclu) if m)
            lignes.append({
                "profil_id": p["profil_id"], "offre_id": o["offre_id"], "echelle": o["echelle"],
                "score_embedding": round(float(c), 4), "score_regles": None if regles is None else round(regles, 4),
                "score_llm": None, "score_final": round(float(final), 4),
                "justification": f"Acquises : {', '.join(acquises) or 'aucune'}. "
                                 f"Manquantes : {', '.join(manquantes) or 'aucune'}."
                                 + (f" Ajustement : {ajustements}" if ajustements else ""),
            })

        if llm_top:
            df_p = pd.DataFrame(lignes)
            index_offres = offres.set_index("offre_id")
            for echelle, groupe in df_p.groupby("echelle"):
                for idx in groupe.nlargest(llm_top, "score_final").index:
                    o = index_offres.loc[lignes[idx]["offre_id"]].to_dict()
                    o["offre_id"] = lignes[idx]["offre_id"]
                    try:
                        j = juger(p, o)
                        lignes[idx]["score_llm"] = max(0, min(100, j.score)) / 100
                        lignes[idx]["justification"] += f" Avis du LLM : {j.justification}"
                    except Exception as e:
                        print(f"  ! jugement LLM {o['offre_id']} : {e}")

        storage.enregistrer_matchings([{k: v for k, v in l.items() if k != "echelle"} for l in lignes])
        resultats.extend(lignes)
        print(f"[Scoring] {p['profil_id']} : {len(lignes)} offres notées")
    return pd.DataFrame(resultats)


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--llm-top", type=int, default=0,
                   help="Nombre d'offres par échelle à faire noter par le LLM (lent : ~10-30 s chacune)")
    storage.init_db()
    df = calculer(a.parse_args().llm_top)
    print(df.groupby("echelle")[["score_embedding", "score_regles", "score_final"]].describe().round(3).T)
