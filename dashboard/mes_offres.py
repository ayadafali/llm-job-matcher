"""Onglet « Mes offres » : importer un CV en PDF et parcourir les offres classées pour ce profil.

Analyse d'un CV = les étapes 1 à 3 du pipeline, pour ce seul fichier :
    PDF -> data/raw/cv/<nom>.pdf -> texte (pdfplumber) -> compétences (LLM local, LM Studio)
        -> scores de toutes les offres extraites (src.scoring)
"""
import contextlib
import html
import io
import json
import re
from collections import Counter

import pandas as pd
import streamlit as st

from dashboard.composants import DRAPEAUX, badge, jauge, puces
from src import aggregation, storage
from src.config import CV_DIR, LLM_MODEL, LM_STUDIO_URL


class ErreurAnalyse(Exception):
    """Message destiné à l'utilisateur, affiché tel quel."""


# --- Analyse d'un CV --------------------------------------------------------------------------------

def nom_profil(texte: str) -> str:
    """'CV Aya Dafali (2026).pdf' -> 'CV_Aya_Dafali_2026' : sert d'identifiant et de nom de fichier."""
    return re.sub(r"[^\w-]+", "_", texte.removesuffix(".pdf").strip()).strip("_")[:60]


def analyser_cv(contenu: bytes, nom: str, llm_top: int, journal) -> None:
    """Enregistre le PDF puis lance ingestion, extraction et scoring. `journal(texte)` affiche l'avancement."""
    from src import extraction, ingestion, llm_client, scoring

    sortie = io.StringIO()   # les modules du pipeline écrivent dans la console : on garde l'interface propre

    journal("Vérification de LM Studio…")
    try:
        llm_client.modeles_disponibles()
    except Exception:
        raise ErreurAnalyse(f"LM Studio ne répond pas ({LM_STUDIO_URL}). Ouvre LM Studio, charge "
                            f"**{LLM_MODEL}**, puis onglet *Developer* → *Start Server*, et recommence.")
    if storage.offres_avec_extraction().empty:
        raise ErreurAnalyse("Aucune offre extraite dans la base : lance d'abord "
                            "`python -m src.extraction --offres`.")

    journal("Lecture du PDF…")
    CV_DIR.mkdir(parents=True, exist_ok=True)
    (CV_DIR / f"{nom}.pdf").write_bytes(contenu)
    storage.enregistrer_profil({"profil_id": nom, "competences": None})   # force une nouvelle extraction
    with contextlib.redirect_stdout(sortie):
        ingestion.ingerer_cv()
    texte = storage.lire("SELECT texte_brut FROM profils WHERE profil_id = ?", (nom,))["texte_brut"].iloc[0]
    if not (texte or "").strip():
        raise ErreurAnalyse("Aucun texte lisible dans ce PDF (CV scanné en image ?). "
                            "Exporte le CV en PDF depuis Word / Google Docs / LaTeX.")

    journal(f"Extraction des compétences par le LLM ({LLM_MODEL})… quelques secondes")
    try:
        with contextlib.redirect_stdout(sortie):
            extraction.extraire_cvs()
    except Exception as e:
        raise ErreurAnalyse(f"Le LLM n'a pas pu lire le CV : {e}")

    journal("Calcul des scores pour toutes les offres… (le premier lancement télécharge "
            "le modèle d'embeddings, compter quelques minutes)")
    try:
        with contextlib.redirect_stdout(sortie):
            scoring.calculer(llm_top)
    except ImportError:
        raise ErreurAnalyse("Le module sentence-transformers manque : `pip install -r requirements.txt`.")


# --- Préparation des données affichées --------------------------------------------------------------

def offres_classees(profil_id: str) -> pd.DataFrame:
    cl = aggregation.classement(profil_id, None, 1_000_000)
    if cl.empty:
        return cl
    ext = storage.offres_avec_extraction()[["offre_id", "competences", "resume", "contrat_extrait", "niveau_exp",
                                            "description", "teletravail", "salaire_annuel", "devise",
                                            "date_publication"]]
    df = cl.merge(ext, on="offre_id", how="left")
    df["contrat"] = df["contrat_extrait"].fillna(df["type_contrat"])
    df["niveau"] = df["niveau_exp"].fillna(df["niveau_experience"])
    df["competences"] = df["competences"].apply(lambda v: v if isinstance(v, list) else [])
    df["rang"] = range(1, len(df) + 1)
    return df


def details(justification: str) -> tuple[str | None, str | None]:
    """Extrait de la justification du scoring : (ajustement appliqué, avis du LLM)."""
    j = justification or ""
    ajust = re.search(r"Ajustement : (.*?)(?= Avis du LLM :|$)", j)
    avis = re.search(r"Avis du LLM : (.*)$", j)
    return (ajust.group(1).strip() if ajust else None), (avis.group(1).strip() if avis else None)


def _salaire(o) -> str | None:
    if pd.isna(o.get("salaire_annuel")):
        return None
    return f"{o['salaire_annuel']:,.0f} {o.get('devise') or ''}/an".replace(",", " ")


# --- Interface --------------------------------------------------------------------------------------

def _import_cv(profils: pd.DataFrame):
    with st.expander("📄 Importer un CV (PDF)", expanded=profils.empty):
        fichier = st.file_uploader("CV au format PDF", type="pdf", key="cv_pdf")
        c1, c2 = st.columns([2, 1])
        nom = nom_profil(c1.text_input("Nom du profil", value=nom_profil(fichier.name) if fichier else "",
                                       help="Identifiant du profil (lettres, chiffres, _ et -)."))
        llm_top = c2.number_input("Avis du LLM (offres)", 0, 10, 0,
                                  help="Le LLM rédige un avis sur les N meilleures offres de chaque échelle. "
                                       "Plus lent : ~10-30 s par offre.")
        if nom and nom in set(profils["profil_id"]):
            st.caption(f"⚠️ Le profil **{nom}** existe déjà : son analyse sera remplacée.")
        if st.button("Analyser mon CV", type="primary", disabled=not (fichier and nom)):
            with st.status("Analyse du CV…", expanded=True) as statut:
                try:
                    analyser_cv(fichier.getvalue(), nom, int(llm_top), statut.write)
                except ErreurAnalyse as e:
                    statut.update(label="L'analyse n'a pas abouti", state="error")
                    st.error(str(e))
                    return
                statut.update(label="CV analysé", state="complete", expanded=False)
            st.session_state["profil_choisi"] = nom
            st.toast(f"Profil {nom} prêt : les offres sont classées ci-dessous.", icon="✅")
            st.rerun()


def _entete_profil(p: pd.Series):
    with st.container(border=True):
        st.markdown(f"**Compétences détectées dans le CV** ({len(p['competences'])})")
        st.markdown(puces(p["competences"], set(p["competences"])), unsafe_allow_html=True)
        exp, form = (json.loads(p.get(c) or "[]") for c in ("experiences", "formation"))
        if exp or form:
            with st.expander("Expériences et formation lues par le LLM"):
                for titre, lignes in (("Expériences", exp), ("Formation", form)):
                    if lignes:
                        st.markdown(f"**{titre}**\n" + "\n".join(f"- {l}" for l in lignes))


def _carte(o, comp_profil: set, texte_cv: str):
    from src.scoring import score_regles

    _, acquises, manquantes = score_regles(o["competences"], comp_profil, texte_cv)
    ajust, avis = details(o["justification"])
    with st.container(border=True):
        gauche, droite = st.columns([5, 1])
        with gauche:
            st.markdown(f"<span style='opacity:.6'>#{o['rang']}</span> &nbsp;"
                        f"<span style='font-size:1.15rem;font-weight:600'>{html.escape(o['titre'] or '')}</span>",
                        unsafe_allow_html=True)
            infos = [o.get("entreprise"), o.get("lieu"), DRAPEAUX.get(o["echelle"])]
            st.caption(" · ".join(str(x) for x in infos if x and not pd.isna(x)))
            etiquettes = [o.get("contrat"), o.get("niveau"), _salaire(o),
                          "Télétravail" if o.get("teletravail") == 1 else None]
            st.markdown("".join(badge(e) for e in etiquettes if e and not pd.isna(e)), unsafe_allow_html=True)
            if o.get("resume"):
                st.markdown(o["resume"])
            st.markdown(puces(acquises + manquantes, set(acquises)), unsafe_allow_html=True)
            if ajust:
                st.caption(f"⚠️ Score réduit : {ajust}")
        with droite:
            st.markdown(jauge(o["score_final"]), unsafe_allow_html=True)
            if o.get("url") and not pd.isna(o["url"]):
                st.link_button("Voir l'offre", o["url"], width="stretch")
        with st.expander("Pourquoi ce score ?"):
            a, b, c = st.columns(3)
            a.metric("Compétences possédées", f"{len(acquises)} / {len(acquises) + len(manquantes)}")
            b.metric("Proximité du texte", f"{o['score_embedding']:.2f}", help="Similarité cosinus des embeddings")
            c.metric("Avis du LLM", "—" if pd.isna(o["score_llm"]) else f"{o['score_llm'] * 100:.0f} / 100")
            if avis:
                st.info(avis)
            if o.get("description") and not pd.isna(o["description"]):
                st.markdown("**Description**")
                st.text((o["description"] or "")[:2500])


def onglet_offres():
    profils = storage.profils()
    _import_cv(profils)
    if profils.empty:
        st.info("Importe un CV ci-dessus pour voir les offres qui lui correspondent.")
        return

    ids = profils["profil_id"].tolist()
    if st.session_state.get("profil_choisi") not in ids:
        st.session_state["profil_choisi"] = ids[0]
    profil_id = st.selectbox("Profil", ids, key="profil_choisi")
    p = profils.set_index("profil_id").loc[profil_id]
    comp_profil, texte_cv = set(p["competences"]), (p["texte_brut"] or "").lower()
    _entete_profil(p)

    df = offres_classees(profil_id)
    if df.empty:
        st.warning("Pas encore de scores pour ce profil.")
        if st.button("Calculer les scores"):
            from src import scoring
            with st.spinner("Calcul des scores…"), contextlib.redirect_stdout(io.StringIO()):
                scoring.calculer()
            st.rerun()
        return

    # Filtres
    c1, c2, c3, c4 = st.columns([2, 2, 3, 1])
    echelle = c1.segmented_control("Échelle", ["toutes", "nationale", "internationale"], default="toutes",
                                   format_func=lambda e: {"toutes": "Toutes", **DRAPEAUX}[e], key="mo_echelle")
    contrats = c2.multiselect("Contrat", sorted(df["contrat"].dropna().unique()), key="mo_contrat",
                              placeholder="Tous")
    recherche = c3.text_input("Rechercher", placeholder="titre, entreprise, compétence…", key="mo_recherche")
    n = c4.selectbox("Afficher", [10, 20, 50, 100], key="mo_n")
    masquer = st.toggle("Masquer les offres pénalisées (niveau trop élevé, habilitation, bénévolat)",
                        key="mo_masquer")

    vue = df
    if echelle and echelle != "toutes":
        vue = vue[vue["echelle"] == echelle]
    if contrats:
        vue = vue[vue["contrat"].isin(contrats)]
    if recherche:
        r = recherche.lower()
        vue = vue[vue.apply(lambda o: r in f"{o['titre']} {o['entreprise']} {' '.join(o['competences'])}".lower(),
                            axis=1)]
    if masquer:
        vue = vue[~vue["justification"].fillna("").str.contains("Ajustement :")]

    if vue.empty:
        st.info("Aucune offre ne correspond à ces filtres.")
        return

    # Ce qui manque le plus parmi les meilleures offres de la sélection
    from src.scoring import score_regles
    tete = vue.head(30)
    manque = Counter(c for _, o in tete.iterrows() for c in score_regles(o["competences"], comp_profil, texte_cv)[2])
    if manque:
        st.markdown(f"**À apprendre en priorité** (parmi les {len(tete)} meilleures offres de la sélection) : "
                    + " ".join(badge(f"{c} · {k} offres") for c, k in manque.most_common(6)),
                    unsafe_allow_html=True)

    st.caption(f"{len(vue)} offres · triées par compatibilité · en bleu : compétences du CV, "
               "en pointillé : compétences à acquérir.")
    for _, o in vue.head(n).iterrows():
        _carte(o, comp_profil, texte_cv)

    c1, c2 = st.columns([1, 3])
    export = vue[["rang", "titre", "entreprise", "lieu", "echelle", "contrat", "niveau", "score_final", "url"]]
    c1.download_button("Télécharger la sélection (CSV)", export.to_csv(index=False, sep=";").encode("utf-8-sig"),
                       file_name=f"offres_{profil_id}.csv", mime="text/csv")
    with c2.expander("Vue tableau"):
        st.dataframe(vue[["rang", "titre", "entreprise", "lieu", "echelle", "contrat", "score_final",
                          "score_regles", "score_embedding", "score_llm", "url"]],
                     hide_index=True, width="stretch",
                     column_config={
                         "score_final": st.column_config.ProgressColumn("Score", min_value=0, max_value=1,
                                                                        format="%.2f"),
                         "url": st.column_config.LinkColumn("Lien"),
                     })
