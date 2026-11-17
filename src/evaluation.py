"""Évaluation : qualité du matching, qualité de l'extraction, comparaison de modèles LM Studio.

1. Préparer les fichiers à remplir à la main (dans eval/, s'ouvrent dans Excel) :
       python -m src.evaluation --preparer
   - eval/annotation_matching.csv : ~30 offres par échelle (les 15 mieux classées + 15 au hasard,
     mélangées, SANS les scores). Colonne « pertinence » : 2 = pertinente, 1 = moyenne, 0 = non pertinente.
   - eval/controle_extraction.csv : 10 offres par échelle avec ce que le LLM a extrait.
     Colonnes ok_competences, ok_niveau, ok_contrat : 1 = correct, 0 = faux (vide = non vérifié).

2. Calculer les métriques une fois les fichiers remplis :
       python -m src.evaluation --calculer
   -> précision@10, nDCG@10, corrélation de Spearman par méthode et par échelle ;
      taux de champs corrects ; statistiques du journal (JSON valides, temps par offre).

3. Comparer des modèles dans LM Studio sur les mêmes offres (sans toucher à la base) :
       python -m src.evaluation --modeles qwen2.5-7b-instruct qwen2.5-3b-instruct --n 30
   -> models/comparaison_modeles.csv. Le premier modèle sert de référence.
"""
import argparse
import json
import math
import time

import numpy as np
import pandas as pd

from src import extraction, llm_client, storage
from src.config import ROOT, SEED

EVAL_DIR = ROOT / "eval"
MODELS_DIR = ROOT / "models"
FICHIER_ANNOTATION = "annotation_matching.csv"
FICHIER_CONTROLE = "controle_extraction.csv"
METHODES = {"score_final": "Score final", "score_regles": "Règles", "score_embedding": "Embeddings",
            "score_llm": "Jugement LLM"}
CSV = dict(sep=";", encoding="utf-8-sig", index=False)   # lisible directement dans Excel (FR)


def _lire_csv(nom):
    return pd.read_csv(EVAL_DIR / nom, sep=";", encoding="utf-8-sig")


def _echantillon(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """n lignes tirées au hasard (reproductible) dans chaque échelle."""
    return pd.concat([g.sample(n=min(n, len(g)), random_state=SEED) for _, g in df.groupby("echelle")])


def _profil_par_defaut() -> str:
    p = storage.profils()
    if p.empty:
        raise RuntimeError("Aucun profil en base.")
    return p["profil_id"].iloc[0]


# --- 1. Préparation -------------------------------------------------------------------------------

def preparer(profil_id: str | None = None, n_top: int = 15, n_hasard: int = 15, forcer: bool = False):
    profil_id = profil_id or _profil_par_defaut()
    EVAL_DIR.mkdir(exist_ok=True)
    cible = EVAL_DIR / FICHIER_ANNOTATION
    if cible.exists() and not forcer:
        print(f"{cible} existe déjà (annotations à préserver). Ajoute --forcer pour le régénérer.")
    else:
        m = storage.lire(
            """SELECT m.offre_id, o.echelle, o.titre, o.entreprise, COALESCE(o.ville, o.region, o.pays) AS lieu,
                      e.niveau_exp AS niveau, COALESCE(e.type_contrat, o.type_contrat) AS contrat,
                      e.competences_requises AS competences_offre, e.resume, o.url, m.score_final
               FROM matchings m JOIN offres o USING (offre_id) JOIN offres_extraites e USING (offre_id)
               WHERE m.profil_id = ?""", (profil_id,))
        rng = np.random.default_rng(SEED)
        morceaux = []
        for _, g in m.groupby("echelle"):
            top = g.nlargest(n_top, "score_final")
            reste = g.drop(top.index)
            hasard = reste.sample(n=min(n_hasard, len(reste)), random_state=SEED)
            morceaux.append(pd.concat([top, hasard]))
        ech = pd.concat(morceaux)
        ech = ech.iloc[rng.permutation(len(ech))].drop(columns="score_final")   # mélangé, sans score
        ech["competences_offre"] = ech["competences_offre"].apply(lambda s: ", ".join(json.loads(s or "[]")))
        ech.insert(0, "profil_id", profil_id)
        ech["pertinence"] = ""
        ech.to_csv(cible, **CSV)
        print(f"{cible.name} : {len(ech)} offres à annoter (2 = pertinente, 1 = moyenne, 0 = non pertinente)")

    cible = EVAL_DIR / FICHIER_CONTROLE
    if cible.exists() and not forcer:
        print(f"{cible} existe déjà. Ajoute --forcer pour le régénérer.")
        return
    e = storage.lire(
        """SELECT e.offre_id, o.echelle, o.titre, SUBSTR(o.description, 1, 1500) AS description_extrait,
                  e.competences_requises AS competences_extraites, e.niveau_exp AS niveau_extrait,
                  e.type_contrat AS contrat_extrait, e.resume AS resume_extrait
           FROM offres_extraites e JOIN offres o USING (offre_id)
           WHERE e.modele NOT LIKE 'source:%'""")
    ctrl = _echantillon(e, 10)
    ctrl["competences_extraites"] = ctrl["competences_extraites"].apply(lambda s: ", ".join(json.loads(s or "[]")))
    for c in ("ok_competences", "ok_niveau", "ok_contrat"):
        ctrl[c] = ""
    ctrl.to_csv(cible, **CSV)
    print(f"{cible.name} : {len(ctrl)} extractions à vérifier (1 = correct, 0 = faux)")


# --- 2. Métriques -----------------------------------------------------------------------------

def precision_at_k(pertinences: list[float], k: int = 10, seuil: int = 2) -> float:
    top = pertinences[:k]
    return float(np.mean([p >= seuil for p in top])) if top else math.nan


def ndcg_at_k(pertinences: list[float], k: int = 10) -> float:
    def dcg(g):
        return sum((2 ** x - 1) / math.log2(i + 2) for i, x in enumerate(g[:k]))
    ideal = dcg(sorted(pertinences, reverse=True))
    return dcg(pertinences) / ideal if ideal > 0 else math.nan


def metriques_matching() -> pd.DataFrame:
    a = _lire_csv(FICHIER_ANNOTATION)
    a = a[pd.to_numeric(a["pertinence"], errors="coerce").notna()].copy()
    if a.empty:
        raise RuntimeError(f"Aucune ligne annotée dans eval/{FICHIER_ANNOTATION} (colonne pertinence).")
    a["pertinence"] = a["pertinence"].astype(float)
    scores = storage.lire("SELECT profil_id, offre_id, score_final, score_regles, score_embedding, score_llm "
                          "FROM matchings")
    df = a.merge(scores, on=["profil_id", "offre_id"], how="left")
    lignes = []
    for echelle, g in df.groupby("echelle"):
        base = {"echelle": echelle, "nb_annotees": len(g),
                "part_pertinentes": round(float((g["pertinence"] >= 2).mean()), 3)}
        for col, nom in METHODES.items():
            s = g.dropna(subset=[col])
            if len(s) < 3:
                continue
            classees = s.sort_values(col, ascending=False)["pertinence"].tolist()
            lignes.append({**base, "methode": nom, "n": len(s),
                           "precision@10": round(precision_at_k(classees), 3),
                           "precision@10 (moyenne incluse)": round(precision_at_k(classees, seuil=1), 3),
                           "ndcg@10": round(ndcg_at_k(classees), 3),
                           "spearman": round(float(s[col].corr(s["pertinence"], method="spearman")), 3)})
    return pd.DataFrame(lignes)


def metriques_extraction() -> pd.DataFrame:
    c = _lire_csv(FICHIER_CONTROLE)
    lignes = []
    for echelle, g in c.groupby("echelle"):
        ligne = {"echelle": echelle}
        for col in ("ok_competences", "ok_niveau", "ok_contrat"):
            v = pd.to_numeric(g[col], errors="coerce").dropna()
            ligne[col.replace("ok_", "correct_")] = round(float(v.mean()), 3) if len(v) else None
            ligne[f"n_{col.replace('ok_', '')}"] = len(v)
        lignes.append(ligne)
    return pd.DataFrame(lignes)


def stats_journal() -> pd.DataFrame:
    """Fiabilité et vitesse du LLM local, d'après le journal d'extraction."""
    return storage.lire(
        """SELECT j.modele, o.echelle,
                  COUNT(*)                                        AS appels,
                  ROUND(100.0 * AVG(j.succes), 1)                 AS pct_succes,
                  ROUND(100.0 * AVG(j.succes = 1 AND j.essais = 1), 1) AS pct_json_valide_1er_essai,
                  ROUND(AVG(j.duree_s), 1)                        AS duree_moy_s,
                  ROUND(AVG(j.nb_competences), 1)                 AS competences_moy
           FROM journal_extraction j LEFT JOIN offres o USING (offre_id)
           GROUP BY j.modele, o.echelle ORDER BY j.modele, o.echelle""")


def calculer():
    EVAL_DIR.mkdir(exist_ok=True)
    for nom, fonction in [("resultats_matching.csv", metriques_matching),
                          ("resultats_extraction.csv", metriques_extraction),
                          ("resultats_journal.csv", stats_journal)]:
        try:
            df = fonction()
        except (FileNotFoundError, RuntimeError) as e:
            print(f"[{nom}] ignoré : {e}")
            continue
        df.to_csv(EVAL_DIR / nom, **CSV)
        print(f"\n=== {nom} ===\n{df.to_string(index=False)}")


# --- 3. Comparaison de modèles dans LM Studio --------------------------------------------------

def jaccard(a: list[str], b: list[str]) -> float:
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 1.0


def _egal(a, b) -> bool:
    """Égalité où deux valeurs absentes (None / NaN) comptent comme identiques."""
    vide = lambda v: v is None or (isinstance(v, float) and math.isnan(v)) or v is pd.NA  # noqa: E731
    return (vide(a) and vide(b)) or (not vide(a) and not vide(b) and a == b)


def comparer_modeles(modeles: list[str], n: int = 30) -> pd.DataFrame:
    offres = storage.lire("SELECT * FROM offres WHERE source != 'data_jobs'")
    ech = _echantillon(offres, n // 2)
    details = []
    for m in modeles:
        print(f"\n[{m}] {len(ech)} offres…")
        for o in ech.to_dict("records"):
            llm_client.dernier_appel.clear()
            t = time.time()
            try:
                x = extraction.extraire_offre(o, modele=m)
                comp, niv, ct, ok = extraction.normaliser_liste(x.competences), x.niveau_experience, x.type_contrat, True
            except Exception:
                comp, niv, ct, ok = [], None, None, False
            details.append({"modele": m, "modele_servi": llm_client.dernier_appel.get("modele_servi"),
                            "offre_id": o["offre_id"], "echelle": o["echelle"], "succes": ok,
                            "json_1er_essai": ok and llm_client.dernier_appel.get("essais") == 1,
                            "duree_s": round(time.time() - t, 2), "competences": comp, "niveau": niv, "contrat": ct})
    d = pd.DataFrame(details).astype({"niveau": object, "contrat": object})
    ref = d[d["modele"] == modeles[0]].set_index("offre_id")
    # Accord calculé seulement sur les offres réussies par le modèle ET par la référence
    comparable = [bool(r["succes"] and ref.loc[r["offre_id"], "succes"]) for _, r in d.iterrows()]
    d["jaccard_vs_ref"] = [jaccard(r["competences"], ref.loc[r["offre_id"], "competences"]) if ok else np.nan
                           for ok, (_, r) in zip(comparable, d.iterrows())]
    d["meme_niveau_que_ref"] = [float(_egal(r["niveau"], ref.loc[r["offre_id"], "niveau"])) if ok else np.nan
                                for ok, (_, r) in zip(comparable, d.iterrows())]
    d["meme_contrat_que_ref"] = [float(_egal(r["contrat"], ref.loc[r["offre_id"], "contrat"])) if ok else np.nan
                                 for ok, (_, r) in zip(comparable, d.iterrows())]
    d["comparable"] = comparable
    d["nb_competences"] = d["competences"].apply(len)
    resume = d.groupby("modele", sort=False).agg(
        offres=("offre_id", "count"), offres_comparees=("comparable", "sum"),
        pct_succes=("succes", "mean"), pct_json_1er_essai=("json_1er_essai", "mean"),
        duree_moy_s=("duree_s", "mean"), competences_moy=("nb_competences", "mean"),
        accord_competences_jaccard=("jaccard_vs_ref", "mean"), accord_niveau=("meme_niveau_que_ref", "mean"),
        accord_contrat=("meme_contrat_que_ref", "mean"),
        modele_reellement_utilise=("modele_servi", lambda v: ", ".join(sorted({str(x) for x in v if pd.notna(x) and x})) or "?"),
    ).round(3).reset_index()
    for _, r in resume.iterrows():
        if r["modele_reellement_utilise"] not in ("?", r["modele"]):
            print(f"ATTENTION : {r['modele']} demandé, mais LM Studio a répondu avec {r['modele_reellement_utilise']}. "
                  "Charge le bon modèle dans LM Studio (ou active le chargement automatique) puis relance.")
    MODELS_DIR.mkdir(exist_ok=True)
    resume.to_csv(MODELS_DIR / "comparaison_modeles.csv", **CSV)
    d.assign(competences=d["competences"].apply(", ".join)).to_csv(MODELS_DIR / "comparaison_modeles_details.csv", **CSV)
    print(f"\n=== Comparaison (référence : {modeles[0]}) ===\n{resume.to_string(index=False)}")
    return resume


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Évaluation du matching, de l'extraction et des modèles")
    p.add_argument("--preparer", action="store_true")
    p.add_argument("--forcer", action="store_true", help="Régénère les fichiers à annoter (efface les annotations)")
    p.add_argument("--profil")
    p.add_argument("--calculer", action="store_true")
    p.add_argument("--modeles", nargs="+", help="Identifiants des modèles LM Studio à comparer")
    p.add_argument("--n", type=int, default=30, help="Nombre d'offres pour la comparaison de modèles")
    a = p.parse_args()
    storage.init_db()
    if a.preparer:
        preparer(a.profil, forcer=a.forcer)
    if a.calculer:
        calculer()
    if a.modeles:
        comparer_modeles(a.modeles, a.n)
