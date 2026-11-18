"""Onglet « Annotation » du dashboard : annoter les fichiers d'évaluation sans passer par Excel.

Lit et écrit exactement les mêmes fichiers que `python -m src.evaluation --preparer`, au même format
(séparateur « ; », UTF-8 avec BOM), donc `python -m src.evaluation --calculer` fonctionne tel quel :
    eval/annotation_matching.csv   -> colonne pertinence : 2 / 1 / 0
    eval/controle_extraction.csv   -> colonnes ok_competences, ok_niveau, ok_contrat : 1 / 0
Chaque clic enregistre immédiatement le fichier (fermer Excel s'il est ouvert dessus).
Raccourcis clavier : 2 / 1 / 0 pour noter, ← / → pour naviguer, N pour la prochaine à annoter.
"""
import contextlib
import html
import io

import pandas as pd
import streamlit as st

from dashboard.composants import LIBELLES, badge, bouton, puces, raccourcis_disponibles
from src import evaluation, storage

PERTINENCES = [("2", "Pertinente"), ("1", "Moyenne"), ("0", "Non pertinente")]
CHAMPS_CONTROLE = {"ok_competences": "Compétences", "ok_niveau": "Niveau", "ok_contrat": "Contrat"}
CHOIX_CONTROLE = {"": "Non vérifié", "1": "Correct", "0": "Faux"}

CONSIGNES_PERTINENCE = """
Se mettre à la place du candidat : **postulerait-il à cette offre, avec des chances raisonnables ?**

| Note | Quand la donner |
|---|---|
| **2 · Pertinente** | Bon métier (data / BI / ML), niveau accessible (stage, débutant, junior), la plupart des compétences clés sont dans le CV. |
| **1 · Moyenne** | Bon métier mais un vrai écart : quelques compétences importantes manquent, niveau un cran au-dessus, ou poste à la marge (ex. contrôle de gestion avec un peu de SQL). |
| **0 · Non pertinente** | Autre métier, niveau senior / manager, habilitation ou condition impossible, ou presque aucune compétence en commun. |

Ne pas tenir compte de la ville ni du salaire. En cas de doute entre deux notes, prendre la plus basse.
Les scores du modèle sont masqués exprès : on juge l'offre, pas le classement.
"""

CONSIGNES_CONTROLE = """
Comparer **l'extrait de l'offre** à ce que le LLM en a tiré, champ par champ :
- **Compétences** : *Correct* si la liste reprend l'essentiel des outils / méthodes demandés, sans en inventer.
  Un oubli mineur reste correct ; une compétence inventée ou plusieurs oublis importants = *Faux*.
- **Niveau** : *Correct* s'il correspond à l'expérience demandée (Débutant 0-1 an, Junior 1-2, Confirmé 3-5,
  Senior 5+), ou s'il est vide alors que l'offre n'en dit rien.
- **Contrat** : *Correct* si le type (CDI, CDD, Stage, Alternance…) est le bon, ou vide si l'offre ne le dit pas.

Si l'extrait est coupé avant l'information, laisser *Non vérifié*.
"""


# --- Lecture / écriture (sans Streamlit, testable) -----------------------------------------------

def chemin(nom: str):
    return evaluation.EVAL_DIR / nom


def charger(nom: str) -> pd.DataFrame | None:
    """Tout en texte : une cellule vide reste vide, les valeurs sont réécrites à l'identique."""
    p = chemin(nom)
    if not p.exists():
        return None
    return pd.read_csv(p, sep=";", encoding="utf-8-sig", dtype=str, keep_default_na=False)


def sauver(df: pd.DataFrame, nom: str):
    df.to_csv(chemin(nom), **evaluation.CSV)


def valeur(v) -> str:
    """'2', '2.0' ou 2 -> '2' ; vide ou illisible -> ''."""
    try:
        return str(int(float(v)))
    except (TypeError, ValueError):
        return ""


def annoter(nom: str, ligne: int, colonnes: dict[str, str]):
    """Écrit les valeurs d'une ligne du fichier (relu juste avant : rien d'autre n'est écrasé)."""
    df = charger(nom)
    for col, v in colonnes.items():
        df.loc[ligne, col] = v
    sauver(df, nom)


def avancement(df: pd.DataFrame, colonnes: list[str]) -> pd.Series:
    """True pour les lignes dont toutes les colonnes d'annotation sont remplies."""
    return df[colonnes].apply(lambda col: col.map(valeur) != "").all(axis=1)


def description_offre(offre_id: str) -> str:
    d = storage.lire("SELECT description FROM offres WHERE offre_id = ?", (offre_id,))
    return "" if d.empty else (d["description"].iloc[0] or "")


# --- Interface ------------------------------------------------------------------------------------

def _liste(s: str) -> list[str]:
    return [c.strip() for c in (s or "").split(",") if c.strip()]


def _barre(df: pd.DataFrame, cle: str, faites: pd.Series, consignes: str):
    """Une seule ligne : consignes, filtres, navigation. Renvoie (offres affichées, position) ou (vide, None).

    La position est gardée dans la session (st.session_state[f"{cle}_pos"]).
    """
    c0, c1, c2, c3, c4, c5, c6 = st.columns([1.1, 1.6, 1.6, 0.6, 0.6, 1.5, 0.9], vertical_alignment="bottom")
    with c0.popover("📋 Consignes", width="stretch"):
        st.markdown(consignes)
    echelles = sorted(df["echelle"].unique())
    choix = c1.selectbox("Échelle", ["toutes"] + echelles, key=f"{cle}_echelle", label_visibility="collapsed",
                         format_func=lambda e: LIBELLES.get(e, "Toutes les échelles"))
    seulement = c2.toggle("Non annotées seulement", key=f"{cle}_restantes")
    vue = df if choix == "toutes" else df[df["echelle"] == choix]
    if seulement:
        vue = vue[~faites.loc[vue.index]]
    if vue.empty:
        return vue, None

    n, cle_pos = len(vue), f"{cle}_pos"
    pos = min(st.session_state.get(cle_pos, 0), n - 1)
    st.session_state[cle_pos] = pos
    a_faire = [p for p, i in enumerate(vue.index) if not faites[i]]
    suivantes = [p for p in a_faire if p > pos] or a_faire

    def aller(p):
        st.session_state[cle_pos] = max(0, min(n - 1, p))

    bouton(c3, "←", "Left", key=f"{cle}_prec", on_click=aller, args=(pos - 1,), disabled=pos == 0,
           width="stretch", help="Précédente")
    bouton(c4, "→", "Right", key=f"{cle}_suiv", on_click=aller, args=(pos + 1,), disabled=pos >= n - 1,
           width="stretch", help="Suivante")
    bouton(c5, "Prochaine à annoter", "N", key=f"{cle}_next", on_click=aller,
           args=(suivantes[0] if suivantes else pos,), disabled=not a_faire, width="stretch")
    c6.markdown(f"<div style='padding-bottom:0.5rem;text-align:right'><b>{pos + 1}</b> / {n}</div>",
                unsafe_allow_html=True)
    return vue, pos


def _progression(df: pd.DataFrame, faites: pd.Series):
    detail = " · ".join(f"{LIBELLES.get(e, e)} {int(faites.loc[g.index].sum())}/{len(g)}"
                        for e, g in df.groupby("echelle"))
    st.progress(float(faites.mean()) if len(df) else 0.0,
                text=f"**Annotées : {int(faites.sum())} / {len(df)}** — {detail}")
    if len(df) and faites.all():
        st.success("Tout est annoté 🎉 Clique sur « Calculer les métriques » en bas de la page.")


def _section_pertinence():
    nom = evaluation.FICHIER_ANNOTATION
    df = charger(nom)
    faites = avancement(df, ["pertinence"])
    _progression(df, faites)
    vue, pos = _barre(df, "pert", faites, CONSIGNES_PERTINENCE)
    if pos is None:
        st.success("Toutes les offres de cette sélection sont annotées.")
        return
    i = vue.index[pos]
    o = df.loc[i]

    profils = storage.profils().set_index("profil_id")
    comp_profil = set(profils.loc[o["profil_id"], "competences"]) if o["profil_id"] in profils.index else set()
    comp_offre = _liste(o.get("competences_offre"))

    actuelle = valeur(o["pertinence"])
    restantes = st.session_state.get("pert_restantes", False)

    def noter(v):
        annoter(nom, i, {"pertinence": v})
        if not restantes:   # sinon l'offre sort de la liste et la suivante prend sa place
            st.session_state["pert_pos"] = min(pos + 1, len(vue) - 1)

    st.markdown("**Cette offre est-elle pertinente pour le profil ?**"
                + (f" — annotation actuelle : **{dict(PERTINENCES)[actuelle]}**" if actuelle else ""))
    cols = st.columns(4)
    for col, (v, libelle) in zip(cols, PERTINENCES):
        bouton(col, f"{v} · {libelle}", v, key=f"pert_{v}", on_click=noter, args=(v,), width="stretch",
               type="primary" if actuelle == v else "secondary")
    bouton(cols[3], "Effacer", "Backspace", key="pert_effacer", on_click=annoter,
           args=(nom, i, {"pertinence": ""}), disabled=not actuelle, width="stretch")
    if raccourcis_disponibles():
        st.caption("⌨️ Clavier : **2 / 1 / 0** pour noter (passe à la suivante), **← →** pour naviguer, "
                   "**N** pour la prochaine non annotée, **Retour arrière** pour effacer.")

    with st.container(border=True):
        st.markdown(f"<div style='font-size:1.3rem;font-weight:600'>{html.escape(o['titre'])}</div>",
                    unsafe_allow_html=True)
        st.caption(" · ".join(x for x in [o.get("entreprise"), o.get("lieu"), LIBELLES.get(o["echelle"])] if x))
        st.markdown(" ".join(badge(x) for x in [f"Niveau : {o.get('niveau') or 'non précisé'}",
                                                 f"Contrat : {o.get('contrat') or 'non précisé'}",
                                                 f"Profil évalué : {o['profil_id']}"]),
                    unsafe_allow_html=True)
        if o.get("resume"):
            st.markdown(o["resume"])
        communes = [x for x in comp_offre if x in comp_profil]
        st.markdown(f"**Compétences demandées** — {len(communes)} / {len(comp_offre)} dans le CV (en bleu)")
        st.markdown(puces(comp_offre, comp_profil), unsafe_allow_html=True)
        d1, d2 = st.columns([3, 1])
        with d1.expander("Description complète de l'offre"):
            st.text(description_offre(o["offre_id"]) or "Description indisponible.")
        if o.get("url"):
            d2.link_button("Ouvrir l'offre ↗", o["url"], width="stretch")



def _section_controle():
    nom = evaluation.FICHIER_CONTROLE
    df = charger(nom)
    faites = avancement(df, list(CHAMPS_CONTROLE))
    _progression(df, faites)
    vue, pos = _barre(df, "ctrl", faites, CONSIGNES_CONTROLE)
    if pos is None:
        st.success("Toutes les extractions de cette sélection sont vérifiées.")
        return
    i = vue.index[pos]
    o = df.loc[i]

    st.markdown(f"<div style='font-size:1.3rem;font-weight:600'>{html.escape(o['titre'])}</div>",
                unsafe_allow_html=True)
    gauche, droite = st.columns([3, 2])
    with gauche:
        st.markdown("**Extrait de l'offre**")
        with st.container(height=320, border=True):
            st.text(o.get("description_extrait") or "")
    with droite, st.form(f"controle_{i}", border=True):
        st.markdown("**Ce que le LLM a extrait**")
        valeurs_lues = {
            "ok_competences": puces(_liste(o.get("competences_extraites")), set()),
            "ok_niveau": html.escape(o.get("niveau_extrait") or "— (vide)"),
            "ok_contrat": html.escape(o.get("contrat_extrait") or "— (vide)"),
        }
        choix = {}
        for col, libelle in CHAMPS_CONTROLE.items():
            st.markdown(f"**{libelle}** : {valeurs_lues[col]}", unsafe_allow_html=True)
            options = list(CHOIX_CONTROLE)
            choix[col] = st.radio(libelle, options, index=options.index(valeur(o[col])), horizontal=True,
                                  format_func=CHOIX_CONTROLE.get, key=f"ctrl_{i}_{col}",
                                  label_visibility="collapsed")
        if o.get("resume_extrait"):
            st.caption(f"Résumé : {o['resume_extrait']}")
        if st.form_submit_button("Enregistrer et passer à la suivante", type="primary", width="stretch"):
            annoter(nom, i, choix)
            if not st.session_state.get("ctrl_restantes", False):
                st.session_state["ctrl_pos"] = min(pos + 1, len(vue) - 1)
            st.rerun()


def _preparation(manquants: list[str]):
    st.info("Fichiers à annoter absents : " + ", ".join(f"`eval/{f}`" for f in manquants) + ". "
            "Ils sont créés à partir des scores en base. Les annotations déjà faites ne sont jamais écrasées.")
    profils = storage.lire("SELECT DISTINCT profil_id FROM matchings ORDER BY profil_id")["profil_id"].tolist()
    if not profils:
        st.warning("Aucun score en base : importe un CV dans l'onglet « Mes offres » (ou lance `python -m src.scoring`).")
        return
    c1, c2 = st.columns([2, 1])
    profil = c1.selectbox("Profil dont on juge les offres", profils, key="ann_profil",
                          help="Les ~30 offres par échelle à noter sont tirées du classement de ce profil.")
    if c2.button("Préparer les fichiers à annoter", type="primary", width="stretch"):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                evaluation.preparer(profil)
        except Exception as e:
            st.error(f"Préparation impossible : {e}")
        else:
            st.rerun()


def onglet_annotation():
    fichiers = [evaluation.FICHIER_ANNOTATION, evaluation.FICHIER_CONTROLE]
    manquants = [f for f in fichiers if not chemin(f).exists()]
    if manquants:
        _preparation(manquants)
        if len(manquants) == len(fichiers):
            return

    disponibles = [f for f in fichiers if f not in manquants]
    titres = {evaluation.FICHIER_ANNOTATION: "① Pertinence des offres",
              evaluation.FICHIER_CONTROLE: "② Contrôle de l'extraction"}
    section = st.segmented_control("Tâche", disponibles, default=disponibles[0], format_func=titres.get,
                                   key="ann_section", label_visibility="collapsed") or disponibles[0]
    if section == evaluation.FICHIER_ANNOTATION:
        _section_pertinence()
    else:
        _section_controle()

    st.divider()
    c1, c2 = st.columns([1, 3])
    if c1.button("Calculer les métriques", width="stretch"):
        sortie = io.StringIO()
        with contextlib.redirect_stdout(sortie):
            evaluation.calculer()
        c2.success("Métriques recalculées : voir l'onglet « Évaluation ».")
        with st.expander("Détail"):
            st.code(sortie.getvalue())
