"""Configuration centralisée du projet."""
from pathlib import Path

# --- Chemins ---
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
CV_DIR = RAW_DIR / "cv"
PROCESSED_DIR = ROOT / "data" / "processed"
DB_PATH = PROCESSED_DIR / "job_matcher.db"

# --- LM Studio (serveur local compatible OpenAI) ---
LM_STUDIO_URL = "http://localhost:1234/v1"
LLM_MODEL = "qwen2.5-7b-instruct"  # identifiant affiché dans LM Studio
LLM_TEMPERATURE = 0.0

# --- Embeddings ---
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"

# --- Filtrage des offres ---
TITRES_CIBLES = (
    r"data analyst|data scientist|data engineer|business intelligence|\bbi\b|"
    r"analytics|machine learning|\bml\b|data science|reporting analyst"
)
NIVEAUX_CIBLES = ["Internship", "Entry level", "Associate"]
TAILLE_ECHANTILLON = 1000
SEED = 42
