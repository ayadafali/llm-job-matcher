"""Onglet « Mes offres » : import d'un CV en PDF -> profil, extraction, scores, offres classées.

LLM et embeddings simulés (comme dans test_pipeline) : ni LM Studio ni téléchargement requis.
"""
import pytest

from dashboard import mes_offres
from src import extraction, ingestion, llm_client, scoring
from tests.test_ingestion import env  # noqa: F401
from tests.test_pipeline import FauxLLM, env_complet, faux_encoder  # noqa: F401


def pdf_texte(texte: str) -> bytes:
    """PDF minimal d'une page contenant `texte` (lisible par pdfplumber)."""
    flux = f"BT /F1 12 Tf 72 720 Td ({texte}) Tj ET".encode()
    objets = [b"<< /Type /Catalog /Pages 2 0 R >>",
              b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
              b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
              b"/Resources << /Font << /F1 5 0 R >> >> >>",
              b"<< /Length %d >>\nstream\n" % len(flux) + flux + b"\nendstream",
              b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out, positions = bytearray(b"%PDF-1.4\n"), []
    for i, o in enumerate(objets, 1):
        positions.append(len(out))
        out += b"%d 0 obj\n" % i + o + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objets) + 1)
    out += b"".join(b"%010d 00000 n \n" % p for p in positions)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objets) + 1, xref)
    return bytes(out)


@pytest.fixture
def base_extraite(env_complet, monkeypatch):  # noqa: F811
    monkeypatch.setattr(llm_client, "chat_json", FauxLLM())
    monkeypatch.setattr(llm_client, "modeles_disponibles", lambda: ["faux-modele"])
    monkeypatch.setattr(scoring, "encoder", faux_encoder)
    monkeypatch.setattr(mes_offres, "CV_DIR", ingestion.CV_DIR)
    ingestion.ingerer("all")
    extraction.extraire_offres()
    return env_complet


def test_import_cv_bout_en_bout(base_extraite):
    etapes = []
    mes_offres.analyser_cv(pdf_texte("Data analyst Python SQL Power BI"), "aya_cv", 0, etapes.append)
    assert (ingestion.CV_DIR / "aya_cv.pdf").exists()
    assert len(etapes) == 4

    df = mes_offres.offres_classees("aya_cv")
    assert len(df) == 7 and df["rang"].tolist() == list(range(1, 8))
    assert df["score_final"].is_monotonic_decreasing
    assert {"competences", "resume", "contrat", "niveau"} <= set(df.columns)
    assert isinstance(df["competences"].iloc[0], list)


def test_import_cv_sans_lm_studio(base_extraite, monkeypatch):
    def hors_ligne():
        raise ConnectionError
    monkeypatch.setattr(llm_client, "modeles_disponibles", hors_ligne)
    with pytest.raises(mes_offres.ErreurAnalyse, match="LM Studio"):
        mes_offres.analyser_cv(pdf_texte("Python"), "x", 0, lambda _: None)


def test_import_pdf_sans_texte(base_extraite):
    with pytest.raises(mes_offres.ErreurAnalyse, match="Aucun texte"):
        mes_offres.analyser_cv(pdf_texte(""), "vide", 0, lambda _: None)


def test_utilitaires():
    assert mes_offres.nom_profil("CV Aya Dafali (2026).pdf") == "CV_Aya_Dafali_2026"
    j = ("Acquises : sql. Manquantes : aws. Ajustement : niveau Senior pour un profil Junior (×0.5). "
         "Avis du LLM : Bon profil.")
    assert mes_offres.details(j) == ("niveau Senior pour un profil Junior (×0.5).", "Bon profil.")
    assert mes_offres.details("Acquises : sql. Manquantes : aucune.") == (None, None)
