"""Client du LLM local servi par LM Studio (API compatible OpenAI).

Avant usage : dans LM Studio, charger le modèle (ex. Qwen2.5-7B-Instruct) puis
onglet Developer -> Start Server (http://localhost:1234 par défaut).

Le modèle n'est PAS entraîné : on l'utilise tel quel, avec un prompt et une sortie JSON imposée
(« structured output »), validée par Pydantic.
"""
import json
import re
import time

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from src.config import LLM_MODEL, LLM_TEMPERATURE, LM_STUDIO_URL

_client = None


def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio", timeout=180)
    return _client


def modeles_disponibles() -> list[str]:
    """Liste les modèles chargés dans LM Studio (vérifie aussi que le serveur répond)."""
    return [m.id for m in client().models.list().data]


def _json_depuis_texte(texte: str) -> dict:
    """Récupère le premier objet JSON d'une réponse, même entourée de ```json ... ```."""
    texte = re.sub(r"^```(?:json)?\s*|\s*```$", "", texte.strip())
    debut, fin = texte.find("{"), texte.rfind("}")
    return json.loads(texte[debut:fin + 1] if debut >= 0 else texte)


# Statistiques du dernier appel (lues par le journal d'extraction) :
# essais = nombre de tentatives (1 = JSON valide du premier coup), succes, duree_s, erreur,
# modele_servi = modèle qui a réellement répondu (renvoyé par LM Studio)
dernier_appel: dict = {}


def chat_json(systeme: str, utilisateur: str, schema: type[BaseModel], essais: int = 3,
              modele: str | None = None) -> BaseModel:
    """Appelle le LLM et renvoie une instance validée de `schema`.

    On demande d'abord une sortie JSON contrainte par le schéma ; si le modèle ou la version de
    LM Studio ne la gère pas, on retombe sur un JSON libre que l'on valide nous-mêmes.
    """
    messages = [{"role": "system", "content": systeme}, {"role": "user", "content": utilisateur}]
    format_strict = {
        "type": "json_schema",
        "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema(), "strict": True},
    }
    derniere_erreur = None
    debut = time.time()
    for essai in range(essais):
        try:
            kwargs = dict(model=modele or LLM_MODEL, messages=messages, temperature=LLM_TEMPERATURE)
            if essai < essais - 1:
                kwargs["response_format"] = format_strict
            rep = client().chat.completions.create(**kwargs)
            resultat = schema.model_validate(_json_depuis_texte(rep.choices[0].message.content))
            dernier_appel.update(essais=essai + 1, succes=True, duree_s=round(time.time() - debut, 2), erreur=None,
                                 modele_servi=getattr(rep, "model", None))
            return resultat
        except (ValidationError, json.JSONDecodeError, ValueError) as e:
            derniere_erreur = e
            time.sleep(0.5)
    dernier_appel.update(essais=essais, succes=False, duree_s=round(time.time() - debut, 2),
                         erreur=str(derniere_erreur)[:300])
    raise RuntimeError(f"Réponse LLM invalide après {essais} essais : {derniere_erreur}")
