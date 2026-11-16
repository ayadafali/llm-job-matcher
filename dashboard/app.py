"""Étape 6 — Dashboard Streamlit.

Lancer depuis la racine du projet :
    streamlit run dashboard/app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src import aggregation, storage  # noqa: E402

# Couleurs : une teinte fixe par échelle (identité), texte en encre neutre, grille discrète.
COULEURS = {"internationale": "#2a78d6", "nationale": "#eb6834"}
LIBELLES = {"internationale": "International (LinkedIn)", "nationale": "France (data_jobs)"}
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
    st.warning("La base est vide : lance d'abord `python -m src.ingestion`.")
    st.stop()

profils = storage.profils()
echelles = [e for e in ("internationale", "nationale") if e in set(vue["echelle"])]

onglets = st.tabs(["Vue d'ensemble", "Compétences demandées", "Matching", "Écart de compétences",
                   "Salaires", "Comparaison des méthodes"])

# --- 1. Vue d'ensemble ---------------------------------------------------------------------------
with onglets[0]:
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
with onglets[1]:
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
                        hovertemplate="%{y} : %{x:.0f} % des offres<extra>" + LIBELLES[e] + "</extra>")
        fig.update_layout(barmode="group", xaxis_title="% des offres qui citent la compétence",
                          legend_traceorder="reversed")
        st.caption("Attention : les compétences internationales sont extraites par le LLM, celles de France "
                   "par le dataset source. Les écarts reflètent aussi cette différence de méthode.")
        st.plotly_chart(style(fig, 40 + 34 * len(comp)), width="stretch")
        with st.expander("Tableau"):
            st.dataframe(comp.iloc[::-1], width="stretch", hide_index=True)

# --- 3. Matching ---------------------------------------------------------------------------------
with onglets[2]:
    if profils.empty:
        st.info("Aucun CV : dépose un PDF dans data/raw/cv/ puis lance `python -m src.ingestion --source cv`.")
    else:
        c1, c2, c3 = st.columns([2, 2, 1])
        profil = c1.selectbox("Profil", profils["profil_id"])
        ech = c2.selectbox("Échelle", ["toutes"] + echelles, format_func=lambda e: LIBELLES.get(e, "Toutes"))
        n = c3.number_input("Offres", 5, 100, 20)
        cl = aggregation.classement(profil, None if ech == "toutes" else ech, n)
        if cl.empty:
            st.info("Pas encore de scores : lance `python -m src.scoring`.")
        else:
            st.dataframe(
                cl[["titre", "entreprise", "lieu", "echelle", "type_contrat", "score_final",
                    "score_regles", "score_embedding", "score_llm", "url"]],
                width="stretch", hide_index=True,
                column_config={
                    "score_final": st.column_config.ProgressColumn("Score final", min_value=0, max_value=1, format="%.2f"),
                    "score_regles": st.column_config.NumberColumn("Règles", format="%.2f"),
                    "score_embedding": st.column_config.NumberColumn("Embeddings", format="%.3f"),
                    "score_llm": st.column_config.NumberColumn("LLM", format="%.2f"),
                    "url": st.column_config.LinkColumn("Lien"),
                })
            st.markdown("#### Détail")
            choix = st.selectbox("Offre", cl.index, format_func=lambda i: f"{cl.loc[i, 'titre']} – {cl.loc[i, 'entreprise']}")
            st.write(cl.loc[choix, "justification"])

# --- 4. Écart de compétences ---------------------------------------------------------------------
with onglets[3]:
    if profils.empty:
        st.info("Aucun CV ingéré.")
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
with onglets[4]:
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
with onglets[5]:
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
