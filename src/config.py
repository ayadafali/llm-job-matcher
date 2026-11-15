"""Configuration centralisée du projet."""
import os
from pathlib import Path

# --- Chemins ---
ROOT = Path(__file__).resolve().parent.parent

try:  # charge les identifiants France Travail depuis .env s'il existe
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass
RAW_DIR = ROOT / "data" / "raw"
CV_DIR = RAW_DIR / "cv"
FT_RAW_DIR = RAW_DIR / "france_travail"      # réponses brutes de l'API, en JSON
PROCESSED_DIR = ROOT / "data" / "processed"
DB_PATH = PROCESSED_DIR / "job_matcher.db"

# --- LM Studio (serveur local compatible OpenAI) ---
LM_STUDIO_URL = "http://localhost:1234/v1"
LLM_MODEL = "qwen2.5-7b-instruct"  # identifiant affiché dans LM Studio
LLM_TEMPERATURE = 0.0

# --- Embeddings (multilingue : offres en anglais ET en français) ---
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"

# --- Échantillonnage commun aux deux échelles ---
TAILLE_ECHANTILLON = 1000   # offres par échelle
SEED = 42

# --- Échelle internationale : LinkedIn (Kaggle) ---
TITRES_CIBLES_EN = (
    r"data analyst|data scientist|data engineer|business intelligence|\bbi\b|"
    r"analytics|machine learning|\bml\b|data science|reporting analyst"
)
NIVEAUX_CIBLES_LINKEDIN = ["Internship", "Entry level", "Associate"]

# --- Échelle nationale : API Offres d'emploi v2 (France Travail) ---
# Identifiants à créer sur https://francetravail.io (application + abonnement à l'API).
# Ne jamais les écrire ici : les mettre dans un fichier .env (ignoré par Git).
FT_CLIENT_ID = os.getenv("FT_CLIENT_ID", "")
FT_CLIENT_SECRET = os.getenv("FT_CLIENT_SECRET", "")
FT_SCOPE = os.getenv("FT_SCOPE", "api_offresdemploiv2 o2dsoffre")
FT_TOKEN_URL = (
    "https://entreprise.francetravail.fr/connexion/oauth2/access_token?realm=%2Fpartenaire"
)
FT_SEARCH_URL = "https://api.francetravail.io/partenaire/offresdemploi/v2/offres/search"
FT_PAGE = 150          # taille max d'une page imposée par l'API
FT_MAX_INDEX = 1149    # dernier index accessible par requête (donc 1 150 offres max)
FT_PAUSE_S = 0.15      # l'API tolère environ 10 appels par seconde

# Une requête par mot-clé ; les résultats sont ensuite dédoublonnés puis filtrés sur le titre.
FT_MOTS_CLES = [
    "data analyst",
    "data scientist",
    "data engineer",
    "analyste de données",
    "business intelligence",
    "machine learning",
    "big data",
    "statisticien",
]
TITRES_CIBLES_FR = (
    r"data|donn[ée]es|business intelligence|\bbi\b|d[ée]cisionnel|machine learning|"
    r"intelligence artificielle|\bia\b|statisticien|big data"
)
