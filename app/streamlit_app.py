"""SmartStay AI: predicción de precios y reglas de negocio.
Ejecutar desde la raíz: python -m streamlit run app/streamlit_app.py
"""
from pathlib import Path
import sys

import joblib
import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SmartStay AI", page_icon="🏠", layout="wide")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.reglas import decidir  # noqa: E402
from src.n8n import enviar_n8n  # noqa: E402

RUTA_MODELO = ROOT / "models" / "mejor_modelo.joblib"

# Coordenadas de referencia aproximadas (centros urbanos representativos).
# El modelo requiere latitud y longitud aunque no se soliciten al usuario.
COORDENADAS_ZONAS = {
    "Barcelona": (41.3874, 2.1686),
    "Girona": (41.9794, 2.8214),
    "Madrid": (40.4168, -3.7038),
    "Málaga": (36.7213, -4.4214),
    "Mallorca": (39.5696, 2.6502),  # Palma
    "Menorca": (39.8885, 4.2658),  # Maó
    "Sevilla": (37.3891, -5.9845),
    "Valencia": (39.4699, -0.3763),
    "Euskadi": (43.2630, -2.9350),  # Bilbao
}

# ==========================================
# ESTILO VISUAL - AZUL METÁLICO
# ==========================================

st.markdown("""
<style>

/* FONDO GENERAL */

.stApp {
    background: linear-gradient(
        135deg,
        #f8fbff 0%,
        #eef4fc 50%,
        #ffffff 100%
    );
    color: #142d59;
}

/* SIDEBAR */

[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #e5efff 0%,
        #f3f7ff 55%,
        #e7effc 100%
    );
    border-right: 1px solid #c2d6f5;
}

[data-testid="stSidebar"] h1 {
    color: #123c80;
    font-weight: 800;
}

[data-testid="stSidebar"] label {
    color: #183d78;
    font-weight: 600;
}

/* SELECTBOX */

.stSelectbox > div > div {
    background-color: #ffffff;
    border: 1px solid #b5ccef;
    border-radius: 10px;
}

/* TÍTULOS */

h1 {
    color: #123b7a !important;
    font-weight: 800 !important;
}

h2, h3 {
    color: #1c4d91 !important;
    font-weight: 700 !important;
}

/* TEXTO */

p {
    color: #344c70;
}

/* SEPARADORES METÁLICOS */

hr {
    border: none !important;
    height: 2px !important;
    background: linear-gradient(
        to right,
        transparent,
        #9bbce9,
        #2365c9,
        #9bbce9,
        transparent
    ) !important;
    margin: 25px 0 !important;
}

/* MÉTRICAS */

[data-testid="stMetric"] {
    background: linear-gradient(
        145deg,
        #ffffff,
        #edf4ff
    );
    border: 1px solid #c4d8f7;
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0 3px 12px rgba(31, 83, 150, 0.07);
}

[data-testid="stMetricLabel"] {
    color: #54749f !important;
}

[data-testid="stMetricValue"] {
    color: #174c9a !important;
    font-weight: 800;
}

/* TARJETAS */

div[data-testid="stAlert"] {
    border-radius: 13px;
    border: 1px solid #bfd4f4;
}

/* BOTONES */

.stButton > button,
.stFormSubmitButton > button {
    background: linear-gradient(
        135deg,
        #367fe0,
        #174b9b
    );
    color: white;
    border: 1px solid #2864b6;
    border-radius: 10px;
    font-weight: 600;
    padding: 9px 22px;
    transition: all 0.25s ease;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: linear-gradient(
        135deg,
        #5599f3,
        #2364bd
    );
    border: 1px solid #174b9b;
    color: white;
    box-shadow: 0 4px 12px rgba(35, 100, 189, 0.25);
}

/* FORMULARIOS */

[data-testid="stForm"] {
    background: linear-gradient(
        145deg,
        #ffffff,
        #f0f5ff
    );
    border: 1px solid #c5d9f6;
    border-radius: 16px;
    padding: 22px;
}

/* CAMPOS DE ENTRADA */

.stNumberInput input {
    border-radius: 9px;
    border-color: #b7cdef;
}

.stSlider [data-baseweb="slider"] {
    color: #286bd0;
}

/* IMÁGENES */

[data-testid="stImage"] img {
    border-radius: 14px;
}

/* CAPTIONS */

.stCaption {
    color: #6c84a7;
}

/* FOOTER LATERAL */

[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #55749d;
}

</style>
""", unsafe_allow_html=True)




def separador_metalico():
    st.markdown("<hr>", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Cargando modelo Random Forest...")
def cargar_modelo(ruta):
    return joblib.load(ruta)


st.sidebar.title("🏠 SmartStay AI")
st.sidebar.markdown("**Sistema inteligente de precios**")
pagina = st.sidebar.selectbox("Navegación", ["➡️ Inicio", "➡️ Evaluación del modelo", "➡️ Nueva predicción"])
st.sidebar.caption("Machine Learning · Reglas de negocio · n8n")

if pagina == "➡️ Inicio":
    st.title("🏠 SmartStay AI")
    st.subheader("Sistema inteligente de recomendación de precios")
    st.write(
        "Estimamos el precio por noche con el Random Forest general optimizado, "
        "entrenado con anuncios de diferentes zonas y rangos de precios, "
        "y aplicamos reglas de negocio para orientar decisiones comerciales."
    )
    st.info("Modelo principal: `models/mejor_modelo.joblib`. La estimación es orientativa y no sustituye una valoración de mercado.")
    separador_metalico()
    a, b, c = st.columns(3)
    a.metric("Problema", "Regresión")
    b.metric("Variable objetivo", "Precio / noche")
    c.metric("Motor de decisión", "11 reglas")
    separador_metalico()
    st.info("Flujo: características del anuncio → predicción → reglas → recomendación. El envío a n8n es opcional y requiere pulsar su botón.")

elif pagina == "➡️ Evaluación del modelo":
    st.title("📊 Evaluación del Random Forest general")
    st.write("Resultados documentados en el conjunto de test reservado (17.923 anuncios).")
    st.subheader("Modelo definitivo · Todo el conjunto de test")
    a, b, c = st.columns(3)
    a.metric("MAE", "100,87 €")
    b.metric("RMSE", "322,47 €")
    c.metric("R²", "0,3973")
    st.caption("17.923 registros; métricas obtenidas en el notebook y verificadas en Streamlit.")
    st.subheader("Análisis complementario · Precio real ≤500 €")
    a, b, c = st.columns(3)
    a.metric("MAE", "54,40 €")
    b.metric("RMSE", "85,91 €")
    c.metric("R²", "0,2895")
    st.caption(
        "16.451 registros. El filtro se aplicó al precio REAL conocido en test "
        "solo para analizar resultados; no se puede aplicar así a anuncios nuevos."
    )
    st.subheader("Comparación de los dos modelos")
    st.dataframe(pd.DataFrame({
        "Modelo": ["General (definitivo)", "Especializado ≤500 € (experimento)"],
        "MAE test completo (€)": [100.87, 115.23],
        "RMSE test completo (€)": [322.47, 401.46],
        "R² test completo": [0.3973, 0.0659],
        "MAE test precio real ≤500 €": [54.40, 49.20],
    }), hide_index=True, use_container_width=True)
    st.info(
        "Se utiliza el modelo general porque ofrece mejores resultados sobre el conjunto completo. "
        "El especializado solo mejora en el subconjunto identificado retrospectivamente por su precio real."
    )
    st.warning("Las métricas son promedios del test, no garantías de precisión para un anuncio individual. No se realizan cambios automáticos de precios.")
    st.caption("Esta página muestra métricas documentadas; no recalcula predicciones ni vuelve a entrenar.")

elif pagina == "➡️ Nueva predicción":
    st.title("🎯 Nueva predicción")
    st.write("Introduce los datos del alojamiento y su precio actual para obtener una recomendación.")
    if not RUTA_MODELO.is_file():
        st.error(f"No se encuentra el modelo en: {RUTA_MODELO}")
        st.stop()
    try:
        modelo = cargar_modelo(str(RUTA_MODELO))
    except Exception as error:
        st.error(f"No se pudo cargar el modelo: {error}")
        st.stop()

    with st.form("formulario_prediccion"):
        st.subheader("Ubicación y tipo de alojamiento")
        c1, c2 = st.columns(2)
        with c1:
            # Las etiquetas deben coincidir exactamente con las del entrenamiento.
            zona = st.selectbox(
                "Selecciona el área del alojamiento",
                options=list(COORDENADAS_ZONAS),
                index=None,
                placeholder="Selecciona una zona...",
            )
            room_type = st.selectbox(
                "Tipo de alojamiento",
                ["Entire home/apt", "Private room", "Shared room", "Hotel room"],
            )
        with c2:
            precio_actual = st.number_input(
                "Precio anunciado actual por noche (€)", min_value=0.01, value=250.0, step=5.0
            )
            minimum_nights = st.number_input(
                "Estancia mínima (noches)", min_value=1, value=2, step=1
            )
        st.caption(
            "La ubicación geográfica se asigna automáticamente a partir del área elegida. "
            "Se utilizan coordenadas aproximadas de referencia, no la dirección exacta del alojamiento."
        )

        separador_metalico()
        st.subheader("Actividad y disponibilidad")
        a, b = st.columns(2)
        with a:
            number_of_reviews = st.number_input("Número total de reseñas", min_value=0, value=20, step=1)
            number_of_reviews_ltm = st.number_input("Reseñas últimos 12 meses", min_value=0, value=1, step=1)
            reviews_per_month = st.number_input("Reseñas por mes", min_value=0.0, value=1.5, step=0.1)
        with b:
            calculated_host_listings_count = st.number_input("Alojamientos del anfitrión", min_value=1, value=3, step=1)
            availability_365 = st.slider("Días disponibles en los próximos 365", 0, 365, 200)
        enviar = st.form_submit_button("Calcular precio y decisión", type="primary", use_container_width=True)

    if enviar:
        # Evita reenviar una predicción anterior si falla el nuevo cálculo.
        st.session_state.pop("datos_n8n", None)
        if zona is None:
            st.error("Selecciona una zona del desplegable.")
            st.stop()
        latitude, longitude = COORDENADAS_ZONAS[zona]
        anuncio = {
            "latitude": latitude,
            "longitude": longitude,
            "minimum_nights": minimum_nights,
            "number_of_reviews": number_of_reviews,
            "reviews_per_month": reviews_per_month,
            "calculated_host_listings_count": calculated_host_listings_count,
            "availability_365": availability_365,
            "number_of_reviews_ltm": number_of_reviews_ltm,
            "zona": zona.strip() if zona is not None else "",
            "room_type": room_type,
            "estancia_larga": float(minimum_nights >= 30),
            "precio_actual": float(precio_actual),
        }
        if not anuncio["zona"]:
            st.error("Selecciona una zona del desplegable.")
            st.stop()
        try:
            # No se incluye precio_actual: es información exclusiva del motor de reglas.
            columnas = [
                "latitude", "longitude", "minimum_nights", "number_of_reviews",
                "reviews_per_month", "calculated_host_listings_count",
                "availability_365", "number_of_reviews_ltm", "zona",
                "room_type", "estancia_larga"
            ]
            X_nuevo = pd.DataFrame([anuncio])[columnas]
            precio_predicho = float(modelo.predict(X_nuevo)[0])
            resultado = decidir(anuncio, precio_predicho)

            # Preparar el envío, sin ejecutarlo hasta pulsar el botón específico.
            precio_sugerido_n8n = resultado.get("precio_sugerido")
            datos_n8n = {
                "zona": zona,
                "tipo_alojamiento": room_type,
                "precio_actual": float(precio_actual),
                "precio_predicho": float(precio_predicho),
                "precio_sugerido": (
                    float(precio_sugerido_n8n)
                    if precio_sugerido_n8n is not None
                    and resultado.get("decision") != "REVISAR PRECIO"
                    else None
                ),
                "decision": resultado["decision"],
                "motivo": resultado["motivo"],
                "reglas_aplicadas": resultado.get("reglas_usadas", []),
                "revision_necesaria": bool(resultado.get("revision_necesaria", False)),
            }
            st.session_state["datos_n8n"] = datos_n8n
        except Exception as error:
            st.error(f"Error al calcular la recomendación: {error}")
            st.stop()

        separador_metalico()
        st.subheader("Resultado de SmartStay AI")
        a, b, c = st.columns(3)
        a.metric("Precio actual", f"{precio_actual:.2f} €")
        b.metric("Estimación IA", f"{precio_predicho:.2f} €")
        c.metric("Decisión de las reglas", resultado["decision"])

        sugerido = resultado.get("precio_sugerido")
        if resultado.get("decision") == "REVISAR PRECIO":
            st.warning("La diferencia supera el 30 % de la estimación. No se propone un precio nuevo; comprueba anuncios comparables.")
            sugerido = None
        if sugerido is not None:
            st.metric(
                "Precio orientativo propuesto (R11)",
                f"{float(sugerido):.2f} €",
                delta=f"{(float(sugerido) / float(precio_actual) - 1) * 100:+.1f}% respecto al actual",
            )
        else:
            st.info("No se propone un nuevo precio: se mantiene el actual o se requiere revisión.")

        st.info(f"**Motivo de la decisión:** {resultado['motivo']}")
        st.write("**Reglas aplicadas:**", " → ".join(resultado.get("reglas_usadas", [])) or "Ninguna")

        if resultado.get("revision_necesaria", False):
            st.warning(
                "**REVISIÓN HUMANA OBLIGATORIA.** "
                + (resultado.get("motivo_revision") or
                   "La decisión debe ser revisada antes de modificar el precio.")
            )
            st.caption("La estimación es orientativa; cualquier propuesta requiere validación antes de aplicarse.")
        elif resultado.get("aplicar_automaticamente", False):
            st.success("Ajuste dentro de los límites definidos. No se ha modificado ningún precio real.")
        else:
            st.info("No se ha indicado ningún cambio automático de precio.")

        st.caption("Estimación orientativa del Random Forest general. Las reseñas recientes no equivalen a reservas y las coordenadas aproximadas pueden reducir la precisión. Ningún precio real se modifica automáticamente.")

    # Botón independiente: no se envía nada al calcular la predicción.
    if "datos_n8n" in st.session_state:
        separador_metalico()
        st.subheader("Automatización con n8n")
        st.caption("Revisa la decisión antes de enviarla al webhook.")
        if st.button("📤 Enviar decisión a n8n", type="primary"):
            try:
                correcto = enviar_n8n(st.session_state["datos_n8n"])
                if correcto:
                    st.success("Decisión enviada correctamente a n8n.")
                    # Evita reenvíos accidentales al volver a pulsar.
                    del st.session_state["datos_n8n"]
                else:
                    st.error("n8n no confirmó la recepción de los datos.")
            except (ValueError, requests.exceptions.RequestException) as error:
                st.error(f"No se pudo enviar la decisión a n8n: {error}")
