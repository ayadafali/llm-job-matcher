"""Onglet Annotation du dashboard : les fichiers écrits restent lisibles par src.evaluation."""
import pandas as pd

from dashboard import annotation
from src import evaluation


def test_aller_retour_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "EVAL_DIR", tmp_path)
    pd.DataFrame({
        "profil_id": ["aya"] * 3, "offre_id": ["li_1", "ft_2", "li_3"],
        "echelle": ["internationale", "nationale", "internationale"],
        "titre": ["Data Analyst", "Data Engineer", "BI"], "competences_offre": ["sql, python", "", "excel"],
        "pertinence": ["", "", "1"],
    }).to_csv(tmp_path / evaluation.FICHIER_ANNOTATION, **evaluation.CSV)

    annotation.annoter(evaluation.FICHIER_ANNOTATION, 0, {"pertinence": "2"})
    df = annotation.charger(evaluation.FICHIER_ANNOTATION)
    assert df["pertinence"].tolist() == ["2", "", "1"]          # la ligne vide reste vide
    assert df["competences_offre"].tolist() == ["sql, python", "", "excel"]
    assert annotation.avancement(df, ["pertinence"]).tolist() == [True, False, True]

    # Relu comme le fait src.evaluation : séparateur ';', BOM, valeurs numériques
    brut = evaluation._lire_csv(evaluation.FICHIER_ANNOTATION)
    assert pd.to_numeric(brut["pertinence"], errors="coerce").tolist()[0] == 2


def test_valeur():
    assert annotation.valeur("2.0") == "2"
    assert annotation.valeur(0) == "0"
    assert annotation.valeur("") == ""
    assert annotation.valeur("x") == ""
