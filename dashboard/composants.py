"""Petits composants d'interface partagés par les onglets du dashboard."""
import html
import inspect

import streamlit as st

LIBELLES = {"internationale": "International (LinkedIn)", "nationale": "France"}
DRAPEAUX = {"internationale": "🌍 International", "nationale": "🇫🇷 France"}

# Onglets « paresseux » (seul l'onglet ouvert est calculé) et raccourcis clavier : versions récentes
# de Streamlit seulement. Sans onglets paresseux, les boutons des onglets cachés existent quand même
# dans la page : un raccourci pourrait alors annoter depuis un autre onglet. On les désactive donc.
_ONGLETS_PARESSEUX = "on_change" in inspect.signature(st.tabs).parameters
_RACCOURCIS = _ONGLETS_PARESSEUX and "shortcut" in inspect.signature(st.button).parameters


def onglets(noms: list[str]):
    if _ONGLETS_PARESSEUX:
        return st.tabs(noms, key="onglet", on_change="rerun")
    return st.tabs(noms)


def ouvert(onglet) -> bool:
    """True si l'onglet est affiché (toujours True sur les anciennes versions de Streamlit)."""
    return bool(getattr(onglet, "open", True)) if _ONGLETS_PARESSEUX else True


def bouton(conteneur, libelle: str, raccourci: str | None = None, **kwargs) -> bool:
    """st.button avec raccourci clavier si la version de Streamlit le permet (ignoré sinon)."""
    if raccourci and _RACCOURCIS:
        kwargs["shortcut"] = raccourci
    return conteneur.button(libelle, **kwargs)


def raccourcis_disponibles() -> bool:
    return _RACCOURCIS


def puces(competences, acquises=(), vide="aucune compétence extraite") -> str:
    """Compétences en pastilles HTML : en bleu celles du profil, en gris pointillé les autres."""
    out = []
    for c in competences:
        if c in acquises:
            style = "background:#2a78d61f;border:1px solid #2a78d6;color:#1d5aa3"
        else:
            style = "background:transparent;border:1px dashed #a8a69f;color:inherit;opacity:.85"
        out.append(f'<span style="display:inline-block;margin:0 6px 6px 0;padding:2px 10px;'
                   f'border-radius:999px;font-size:0.85rem;{style}">{html.escape(str(c))}</span>')
    return "".join(out) or f"<em>{vide}</em>"


def badge(texte: str) -> str:
    return (f'<span style="display:inline-block;margin:0 6px 4px 0;padding:1px 8px;border-radius:6px;'
            f'background:#8881;font-size:0.8rem">{html.escape(str(texte))}</span>')


def jauge(score: float) -> str:
    """Score 0-1 affiché en pourcentage avec une barre, couleur selon le niveau."""
    pct = max(0, min(100, round(score * 100)))
    couleur = "#2f9e5b" if pct >= 60 else "#d39b1f" if pct >= 35 else "#a8a69f"
    return (f'<div style="text-align:right"><div style="font-size:1.8rem;font-weight:700;line-height:1">{pct}'
            f'<span style="font-size:1rem;font-weight:500"> %</span></div>'
            f'<div style="height:6px;border-radius:3px;background:#8882;margin-top:6px">'
            f'<div style="width:{pct}%;height:6px;border-radius:3px;background:{couleur}"></div></div>'
            f'<div style="font-size:0.75rem;opacity:.7;margin-top:2px">compatibilité</div></div>')
