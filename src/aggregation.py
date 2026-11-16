"""Étape 5 — Indicateurs marché et gap analysis, par échelle (internationale / nationale).

Les salaires ne sont comparés qu'à l'intérieur d'une même échelle (devises et années différentes).
"""
import pandas as pd

from src import storage


def vue_ensemble() -> pd.DataFrame:
    return storage.lire(
        """
        SELECT o.echelle, o.source,
               COUNT(*)                                    AS nb_offres,
               SUM(e.offre_id IS NOT NULL)                 AS nb_extraites,
               ROUND(100.0 * AVG(o.salaire_annuel IS NOT NULL), 1) AS pct_salaire,
               ROUND(100.0 * AVG(o.teletravail = 1), 1)    AS pct_teletravail,
               MIN(o.date_publication) AS debut, MAX(o.date_publication) AS fin
        FROM offres o LEFT JOIN offres_extraites e USING (offre_id)
        GROUP BY o.echelle, o.source ORDER BY o.echelle
        """
    )


def top_competences(echelle: str, n: int = 15) -> pd.DataFrame:
    """Compétences les plus demandées : nombre d'offres et part des offres extraites qui les citent."""
    df = storage.offres_avec_extraction(echelle)
    if df.empty:
        return pd.DataFrame(columns=["competence", "nb_offres", "part"])
    comp = df.explode("competences").dropna(subset=["competences"])
    out = comp.groupby("competences")["offre_id"].nunique().sort_values(ascending=False).head(n)
    return pd.DataFrame({"competence": out.index, "nb_offres": out.values,
                         "part": (out.values / len(df)).round(3)})


def comparaison_competences(n: int = 15) -> pd.DataFrame:
    """Part des offres citant chaque compétence, France vs international (union des tops n)."""
    inter = top_competences("internationale", 200).set_index("competence")["part"]
    nat = top_competences("nationale", 200).set_index("competence")["part"]
    tops = list(dict.fromkeys(list(inter.head(n).index) + list(nat.head(n).index)))
    df = pd.DataFrame({"competence": tops,
                       "internationale": [inter.get(c, 0.0) for c in tops],
                       "nationale": [nat.get(c, 0.0) for c in tops]})
    df["ecart"] = df["nationale"] - df["internationale"]
    return df.sort_values("internationale", ascending=False).reset_index(drop=True)


def repartition(colonne: str, echelle: str) -> pd.DataFrame:
    autorisees = {"type_contrat", "niveau_experience", "region", "source"}
    if colonne not in autorisees:
        raise ValueError(colonne)
    return storage.lire(
        f"SELECT COALESCE({colonne}, 'Non précisé') AS valeur, COUNT(*) AS nb FROM offres "
        f"WHERE echelle = ? GROUP BY valeur ORDER BY nb DESC", (echelle,)
    )


def salaires(echelle: str) -> pd.DataFrame:
    """Salaire annuel médian par niveau, dans la devise de l'échelle (USD ou EUR)."""
    df = storage.lire(
        "SELECT COALESCE(niveau_experience, 'Non précisé') AS niveau, devise, salaire_annuel "
        "FROM offres WHERE echelle = ? AND salaire_annuel IS NOT NULL", (echelle,)
    )
    if df.empty:
        return df
    return (df.groupby(["niveau", "devise"])["salaire_annuel"]
              .agg(nb="count", median="median", q1=lambda s: s.quantile(.25), q3=lambda s: s.quantile(.75))
              .round(0).reset_index())


def classement(profil_id: str, echelle: str | None = None, n: int = 20) -> pd.DataFrame:
    sql = """
        SELECT m.offre_id, o.echelle, o.titre, o.entreprise, COALESCE(o.ville, o.region, o.pays) AS lieu,
               o.type_contrat, o.niveau_experience, o.url,
               m.score_final, m.score_regles, m.score_embedding, m.score_llm, m.justification
        FROM matchings m JOIN offres o USING (offre_id)
        WHERE m.profil_id = ?"""
    params = [profil_id]
    if echelle:
        sql += " AND o.echelle = ?"
        params.append(echelle)
    sql += " ORDER BY m.score_final DESC LIMIT ?"
    params.append(n)
    return storage.lire(sql, params)


def gap(profil_id: str, echelle: str, n: int = 20) -> pd.DataFrame:
    """Compétences les plus demandées du marché, et si le profil les possède."""
    top = top_competences(echelle, n)
    p = storage.profils().set_index("profil_id").loc[profil_id]
    texte = (p["texte_brut"] or "").lower()
    from src.scoring import present_dans_cv
    top["possedee"] = [c in set(p["competences"]) or present_dans_cv(c, texte) for c in top["competence"]]
    return top


def taux_couverture(profil_id: str, echelle: str, n: int = 20) -> float:
    """Part (pondérée par la demande) des n compétences les plus demandées que le profil possède."""
    g = gap(profil_id, echelle, n)
    return 0.0 if g.empty else round(float((g["part"] * g["possedee"]).sum() / g["part"].sum()), 3)


def comparaison_methodes() -> pd.DataFrame:
    """Corrélation entre les scores des trois méthodes (Spearman), par échelle."""
    df = storage.lire("SELECT o.echelle, m.score_embedding, m.score_regles, m.score_llm, m.score_final "
                      "FROM matchings m JOIN offres o USING (offre_id)")
    out = []
    for echelle, g in df.groupby("echelle"):
        c = g[["score_embedding", "score_regles", "score_llm"]].corr(method="spearman")
        out.append({"echelle": echelle,
                    "embeddings_vs_regles": round(c.loc["score_embedding", "score_regles"], 3),
                    "llm_vs_regles": round(c.loc["score_llm", "score_regles"], 3) if g["score_llm"].notna().sum() > 2 else None,
                    "llm_vs_embeddings": round(c.loc["score_llm", "score_embedding"], 3) if g["score_llm"].notna().sum() > 2 else None})
    return pd.DataFrame(out)


if __name__ == "__main__":
    print(vue_ensemble().to_string(index=False))
    print("\nCompétences France vs international :")
    print(comparaison_competences().to_string(index=False))
