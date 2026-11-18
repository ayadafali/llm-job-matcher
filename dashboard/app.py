"""Étape 6 — Dashboard Streamlit.

Lancer depuis la racine du projet :
    streamlit run dashboard/app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard.composants import onglets, ouvert  # noqa: E402
from src import aggregation, storage  # noqa: E402

# Couleurs : une teinte fixe par échelle (identité), texte en encre neutre, grille discrète.
COULEURS = {"internationale": "#2a78d6", "nationale": "#eb6834"}
LIBELLES = {"internationale": "International (LinkedIn)", "nationale": "France"}
NOMS_SOURCES = {"france_travail": "France Travail", "data_jobs": "data_jobs", "linkedin": "LinkedIn"}
NEUTRE, ENCRE, ENCRE_2, GRILLE = "#a8a69f", "#0b0b0b", "#52514e", "#e8e7e3"


def style(fig, hauteur=420):
    fig.update_layout(
        height=hauteur, template="plotly_white", margin=dict(l=10, r=10, t=40, b=10),
        font=dict(color=ENCRE_2, size=13), legend=dict(orientation="h", y=1.08, x=0),
        bargap=0.25, bargroupgap=0.08, hoverlabel=dict(bgcolor="white", font_color=ENCRE),
    )
    fig.update_xaxes(gridcolor=GRILLE, zeroline=False)
    fig.update_yaxes(gridcolor=GRILLE, zeroline=False)
    return fig


st.set_page_config(page_title="LLM Job Matcher", layout="wide")
st.title("LLM Job Matcher")
st.caption("Matching CV / offres d'emploi avec un LLM local, sur deux échelles : internationale et France.")

try:
    storage.init_db()
    vue = aggregation.vue_ensemble()
except Exception as e:
    st.error(f"Base de données indisponible : {e}")
    st.stop()
if vue.empty:
    st.warning("La base est vide. Deux possibilités :\n"
               "- récupérer le fichier `job_matcher.db` d'un membre de l'équipe et le copier dans "
               "`data/processed/` (le plus simple : il n'est pas dans Git) ;\n"
               "- ou le reconstruire avec le pipeline (`python -m src.ingestion`, puis extraction et scoring, "
               "voir le README).")
    st.stop()

sources_nat = vue.loc[vue["echelle"] == "nationale", "source"].tolist()
if sources_nat:
    LIBELLES["nationale"] = "France (" + ", ".join(NOMS_SOURCES.get(s, s) for s in sources_nat) + ")"

profils = storage.profils()
echelles = [e for e in ("internationale", "nationale") if e in set(vue["echelle"])]

NOMS_ONGLETS = ["Mes offres", "Annotation", "Vue d'ensemble", "Compétences demandées", "Écart de compétences",
                "Salaires", "Comparaison des méthodes", "Évaluation"]
ONGLETS = dict(zip(NOMS_ONGLETS, onglets(NOMS_ONGLETS)))

# --- 0. Mes offres : import d'un CV et offres classées ---------------------------------------------
with ONGLETS["Mes offres"] as t:
    if ouvert(t):
        from dashboard.mes_offres import onglet_offres

        onglet_offres()

# --- 1. Vue d'ensemble ---------------------------------------------------------------------------
with ONGLETS["Vue d'ensemble"] as t:
    if ouvert(t):
        cols = st.columns(len(echelles))
        for col, e in zip(cols, echelles):
            v = vue[vue["echelle"] == e]
            with col:
                st.subheader(LIBELLES[e])
                a, b, c = st.columns(3)
                a.metric("Offres", int(v["nb_offres"].sum()))
                b.metric("Extraites", int(v["nb_extraites"].sum()))
                c.metric("Avec salaire", f"{v['pct_salaire'].iloc[0]:.0f} %")
                st.caption(f"Période : {v['debut'].iloc[0]} → {v['fin'].iloc[0]}")
        st.markdown("#### Répartition des offres")
        choix = st.radio("Critère", ["type_contrat", "niveau_experience", "region"], horizontal=True,
                         format_func={"type_contrat": "Type de contrat", "niveau_experience": "Niveau",
                                      "region": "Région / État"}.get)
        cols = st.columns(len(echelles))
        for col, e in zip(cols, echelles):
            r = aggregation.repartition(choix, e).head(12).iloc[::-1]
            fig = go.Figure(go.Bar(x=r["nb"], y=r["valeur"], orientation="h", marker_color=COULEURS[e],
                                   hovertemplate="%{y} : %{x} offres<extra></extra>"))
            fig.update_layout(title=LIBELLES[e])
            col.plotly_chart(style(fig, 380), width="stretch")

# --- 2. Compétences demandées --------------------------------------------------------------------
with ONGLETS["Compétences demandées"] as t:
    if ouvert(t):
        n = st.slider("Nombre de compétences par échelle", 5, 25, 12)
        comp = aggregation.comparaison_competences(n)
        if comp.empty:
            st.info("Aucune offre extraite : lance `python -m src.extraction --offres`.")
        else:
            comp = comp.iloc[::-1]
            fig = go.Figure()
            for e in reversed(echelles):   # en barres horizontales, la première série s'affiche en bas
                fig.add_bar(y=comp["competence"], x=comp[e] * 100, name=LIBELLES[e], orientation="h",
                            marker_color=COULEURS[e],
                            error_x=dict(type="data", array=comp[f"ic95_{e}"] * 100, color=NEUTRE, thickness=1.2, width=3),
                            hovertemplate="%{y} : %{x:.0f} % des offres<extra>" + LIBELLES[e] + "</extra>")
            fig.update_layout(barmode="group", xaxis_title="% des offres qui citent la compétence",
                              legend_traceorder="reversed")
            if "data_jobs" in sources_nat:
                st.caption("Attention : les compétences internationales sont extraites par le LLM, celles de France "
                           "par le dataset source. Les écarts reflètent aussi cette différence de méthode.")
            else:
                st.caption("Compétences extraites par le même LLM local pour les deux échelles.")
            st.caption("Traits gris : intervalle de confiance à 95 %. Écarts France / international "
                       "statistiquement significatifs : "
                       + (", ".join(comp.loc[comp["ecart_significatif"], "competence"].iloc[::-1]) or "aucun") + ".")
            st.plotly_chart(style(fig, 40 + 34 * len(comp)), width="stretch")
            with st.expander("Tableau"):
                st.dataframe(comp.iloc[::-1], width="stretch", hide_index=True)

# --- 4. Écart de compétences ---------------------------------------------------------------------
with ONGLETS["Écart de compétences"] as t:
    if ouvert(t):
        if profils.empty:
            st.info("Aucun CV : importe-le dans l'onglet « Mes offres ».")
        else:
            c1, c2 = st.columns(2)
            profil = c1.selectbox("Profil ", profils["profil_id"])
            ech = c2.selectbox("Échelle ", echelles, format_func=LIBELLES.get)
            g = aggregation.gap(profil, ech, 20)
            if g.empty:
                st.info("Aucune offre extraite pour cette échelle.")
            else:
                st.metric("Couverture des 20 compétences les plus demandées (pondérée par la demande)",
                          f"{aggregation.taux_couverture(profil, ech) * 100:.0f} %")
                g = g.iloc[::-1]
                fig = go.Figure()
                for possede, nom, couleur in [(True, "Possédée", COULEURS[ech]), (False, "À acquérir", NEUTRE)]:
                    s = g[g["possedee"] == possede]
                    fig.add_bar(y=s["competence"], x=s["part"] * 100, orientation="h", name=nom, marker_color=couleur,
                                hovertemplate="%{y} : %{x:.0f} % des offres<extra>" + nom + "</extra>")
                fig.update_layout(barmode="overlay", xaxis_title="% des offres qui citent la compétence",
                                  yaxis=dict(categoryorder="array", categoryarray=list(g["competence"])))
                st.plotly_chart(style(fig, 40 + 26 * len(g)), width="stretch")
                manquantes = g[~g["possedee"]].iloc[::-1]["competence"].head(5).tolist()
                if manquantes:
                    st.success("Compétences à développer en priorité : " + ", ".join(manquantes))

# --- 5. Salaires ---------------------------------------------------------------------------------
with ONGLETS["Salaires"] as t:
    if ouvert(t):
        st.caption("Salaires annuels bruts, comparés uniquement à l'intérieur d'une échelle "
                   "(devises et années de publication différentes).")
        cols = st.columns(len(echelles))
        for col, e in zip(cols, echelles):
            s = aggregation.salaires(e)
            col.subheader(LIBELLES[e])
            if s.empty:
                col.info("Aucun salaire renseigné.")
            else:
                col.dataframe(s, hide_index=True, width="stretch")

# --- 6. Comparaison des méthodes -----------------------------------------------------------------
with ONGLETS["Comparaison des méthodes"] as t:
    if ouvert(t):
        st.caption("Corrélation de Spearman entre les scores des trois méthodes de matching.")
        st.dataframe(aggregation.comparaison_methodes(), hide_index=True, width="stretch")
        if not profils.empty:
            m = storage.lire("SELECT o.echelle, o.titre, m.score_embedding, m.score_regles FROM matchings m "
                             "JOIN offres o USING (offre_id) WHERE m.profil_id = ? AND m.score_regles IS NOT NULL",
                             (profils["profil_id"].iloc[0],))
            fig = go.Figure()
            for e in echelles:
                s = m[m["echelle"] == e]
                fig.add_scatter(x=s["score_regles"], y=s["score_embedding"], mode="markers", name=LIBELLES[e],
                                marker=dict(color=COULEURS[e], size=8, opacity=.7, line=dict(color="white", width=1)),
                                text=s["titre"], hovertemplate="%{text}<br>Règles %{x:.2f} · Embeddings %{y:.3f}<extra></extra>")
            fig.update_layout(xaxis_title="Score règles (part des compétences possédées)",
                              yaxis_title="Similarité embeddings (cosinus)")
            st.plotly_chart(style(fig, 460), width="stretch")

# --- 7. Évaluation ---------------------------------------------------------------------------------
with ONGLETS["Évaluation"] as t:
    if ouvert(t):
        from src.evaluation import EVAL_DIR, MODELS_DIR, stats_journal

        def afficher_csv(chemin, titre, vide):
            st.markdown(f"#### {titre}")
            try:
                df = pd.read_csv(chemin, sep=";", encoding="utf-8-sig")
            except (FileNotFoundError, pd.errors.EmptyDataError):   # fichier vide : pas assez d'annotations
                df = pd.DataFrame()
            if df.empty:
                st.info(vide + " (au moins 3 offres annotées par échelle)" if chemin.exists() else vide)
            else:
                st.dataframe(df, hide_index=True, width="stretch")

        afficher_csv(EVAL_DIR / "resultats_matching.csv", "Qualité du matching (offres annotées à la main)",
                     "Annoter les offres dans l'onglet « Annotation » puis cliquer sur « Calculer les métriques ».")
        st.caption("Précision@10 : part d'offres pertinentes parmi les 10 premières selon la méthode. "
                   "nDCG@10 : qualité de l'ordre (1 = ordre idéal). Spearman : accord entre score et annotation.")
        afficher_csv(EVAL_DIR / "resultats_extraction.csv", "Qualité de l'extraction (vérification manuelle)",
                     "Vérifier les extractions dans l'onglet « Annotation » puis cliquer sur « Calculer les métriques ».")
        st.markdown("#### Fiabilité et vitesse du LLM local (journal d'extraction)")
        j = stats_journal()
        if j.empty:
            st.info("Le journal se remplit lors des prochaines extractions.")
        else:
            st.dataframe(j, hide_index=True, width="stretch")
        afficher_csv(MODELS_DIR / "comparaison_modeles.csv", "Comparaison de modèles dans LM Studio",
                     "Lancer `python -m src.evaluation --modeles <modèle 1> <modèle 2> --n 30`.")

# --- 8. Annotation -------------------------------------------------------------------------------
with ONGLETS["Annotation"] as t:
    if ouvert(t):
        from dashboard.annotation import onglet_annotation

        onglet_annotation()
