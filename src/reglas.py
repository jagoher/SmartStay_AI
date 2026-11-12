"""Motor de reglas de SmartStay AI para el Random Forest general definitivo.

Modelo: models/mejor_modelo.joblib (sin filtro de precio <= 500 EUR).
La función decidir mantiene la interfaz utilizada por Streamlit y n8n.

R1-R10: clasificación comercial orientativa.
R11: control de discrepancias, sin ajuste porcentual fijo.

La estimación ML no es un precio de mercado verificado.
"""
from __future__ import annotations

from math import isfinite
from typing import Any, Mapping

# Columnas utilizadas por el Random Forest general entrenado.
VARIABLES_MODELO = (
    "latitude", "longitude", "minimum_nights", "number_of_reviews",
    "reviews_per_month", "calculated_host_listings_count",
    "availability_365", "number_of_reviews_ltm",
    "zona", "room_type", "estancia_larga",
)


def decidir(anuncio: Mapping[str, Any], precio_predicho: float) -> dict[str, Any]:
    """Devuelve una recomendación orientativa compatible con Streamlit.

    Se mantiene el contrato: decision, motivo, reglas_usadas,
    precio_sugerido, revision_necesaria, motivo_revision y
    aplicar_automaticamente.
    """
    resultado: dict[str, Any] = {
        "decision": "REVISAR",
        "motivo": "No hay información suficiente para recomendar un cambio.",
        "reglas_usadas": [],
        "precio_sugerido": None,
        "revision_necesaria": True,
        "motivo_revision": "Se requiere comprobación humana.",
        "aplicar_automaticamente": False,
    }

    try:
        actual = float(anuncio.get("precio_actual"))
        estimado = float(precio_predicho)
        disponibilidad = int(anuncio.get("availability_365", 0))
        resenas_recientes = int(anuncio.get("number_of_reviews_ltm", 0))
    except (ValueError, TypeError, OverflowError):
        resultado.update(motivo="Datos numéricos ausentes o inválidos.", reglas_usadas=["R1"])
        return resultado

    if (not isfinite(actual) or not isfinite(estimado) or actual <= 0 or
            estimado <= 0 or not 0 <= disponibilidad <= 365 or resenas_recientes < 0):
        resultado.update(motivo="Datos fuera de rango o inválidos.", reglas_usadas=["R1"])
        return resultado

    if disponibilidad == 0:
        resultado.update(
            decision="SIN DISPONIBILIDAD",
            motivo="El anuncio no tiene días disponibles para nuevas reservas.",
            reglas_usadas=["R2"],
            revision_necesaria=False,
            motivo_revision=None,
        )
        return resultado

    ratio = actual / estimado
    diferencia_relativa = abs(actual - estimado) / estimado
    if ratio < 0.85:
        decision = "SUBIR"
        motivo = "El precio anunciado está por debajo de la estimación del modelo."
        reglas = ["R3", "R6"]
    elif ratio <= 1.15:
        decision = "MANTENER"
        motivo = "El precio anunciado es próximo a la estimación del modelo."
        reglas = ["R4", "R7"]
    else:
        reglas = ["R5"]
        if resenas_recientes >= 10:
            decision = "MANTENER"
            motivo = "Precio superior a la estimación, con actividad reciente elevada."
            reglas.append("R8")
        elif resenas_recientes <= 2:
            decision = "BAJAR"
            motivo = "Precio superior a la estimación y pocas reseñas recientes."
            reglas.append("R9")
        else:
            decision = "REVISAR"
            motivo = "Precio superior a la estimación; conviene revisar la estrategia."
            reglas.append("R10")

    # R11: la distancia al modelo NO es una medida de incertidumbre estadística.
    # No se aplica ningún porcentaje fijo ni se automatizan cambios de precio.
    reglas.append("R11")
    if diferencia_relativa > 0.30:
        resultado.update(
            decision="REVISAR PRECIO",
            motivo=motivo + " La discrepancia supera el 30 % de la estimación.",
            reglas_usadas=reglas,
            precio_sugerido=None,
            revision_necesaria=True,
            motivo_revision=(
                "Diferencia importante entre precio anunciado y estimación. "
                "Contrastar alojamientos comparables antes de cambiar el precio."
            ),
        )
    elif diferencia_relativa > 0.15:
        resultado.update(
            decision=decision,
            motivo=motivo,
            reglas_usadas=reglas,
            precio_sugerido=round(estimado, 2) if decision in ("SUBIR", "BAJAR") else None,
            revision_necesaria=True,
            motivo_revision=(
                "La diferencia supera el 15 %: el precio estimado es solo una "
                "referencia y requiere validación humana."
            ),
        )
    else:
        resultado.update(
            decision=decision,
            motivo=motivo,
            reglas_usadas=reglas,
            precio_sugerido=round(estimado, 2) if decision in ("SUBIR", "BAJAR") else None,
            revision_necesaria=False,
            motivo_revision=None,
        )
    return resultado


def predecir_y_decidir(anuncio: Mapping[str, Any], modelo: Any) -> dict[str, Any]:
    """Predice con el modelo general y aplica R1-R11 sin filtrar por precio.

    `modelo` es el objeto cargado desde `models/mejor_modelo.joblib`.
    `anuncio` contiene las 11 variables de entrada y `precio_actual`.
    `estancia_larga` debe tener la misma codificación que en el entrenamiento.
    Devuelve los mismos campos que `decidir`, más `precio_predicho`.

    No entrena ni modifica el modelo, y nunca aplica cambios automáticos.
    """
    import pandas as pd

    datos = dict(anuncio)
    faltantes = [col for col in VARIABLES_MODELO if col not in datos]
    if faltantes:
        raise ValueError(f"Faltan variables del modelo: {', '.join(faltantes)}")

    entrada = pd.DataFrame([{col: datos[col] for col in VARIABLES_MODELO}])
    prediccion = float(modelo.predict(entrada)[0])
    resultado = decidir(anuncio, prediccion)
    return {"precio_predicho": round(prediccion, 2), **resultado}
