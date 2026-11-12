import os
import requests

from pathlib import Path
from dotenv import load_dotenv

# Cargar el archivo .env desde la raíz del proyecto
RUTA_ENV = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(RUTA_ENV)

# Variables de entorno
N8N_URL = os.getenv("N8N_URL")
N8N_CABECERA = os.getenv("N8N_CABECERA", "X-Api-Key")
N8N_CLAVE = os.getenv("N8N_CLAVE")


def enviar_n8n(datos):
    """
    Envía la decisión de SmartStay AI al webhook de n8n.

    Args:
        datos (dict): Información del alojamiento,
                      predicción y decisión.

    Returns:
        bool: True si n8n acepta la petición.
    """

    if not N8N_URL or not N8N_CLAVE:
        raise ValueError(
            "Faltan N8N_URL o N8N_CLAVE en el archivo .env"
        )

    cabeceras = {
        N8N_CABECERA: N8N_CLAVE,
        "Content-Type": "application/json"
    }

    respuesta = requests.post(
        N8N_URL,
        json=datos,
        headers=cabeceras,
        timeout=10
    )

    respuesta.raise_for_status()

    return 200 <= respuesta.status_code < 300