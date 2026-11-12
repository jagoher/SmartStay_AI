
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RAIZ = Path(__file__).resolve().parent.parent

# Cargar el modelo definitivo y el test
modelo = joblib.load(RAIZ / "models" / "mejor_modelo.joblib")

X_test = pd.read_csv(RAIZ / "data" / "processed" / "X_test.csv")
y_test = pd.read_csv(RAIZ / "data" / "processed" / "y_test.csv")["price"]

# Predicciones sin modificar el modelo
predicciones = modelo.predict(X_test)

# Crear tabla de análisis
df = X_test.copy()
df["precio_real"] = y_test.to_numpy()
df["precio_predicho"] = predicciones
df["error_absoluto"] = abs(df["precio_real"] - df["precio_predicho"])
df["error_cuadrado"] = (df["precio_real"] - df["precio_predicho"]) ** 2

def resumen(grupo):
    return pd.Series({
        "alojamientos": len(grupo),
        "precio_medio": grupo["precio_real"].mean(),
        "MAE": grupo["error_absoluto"].mean(),
        "RMSE": np.sqrt(grupo["error_cuadrado"].mean())
    })

print("\n" + "=" * 60)
print("1. ERRORES POR TIPO DE ALOJAMIENTO")
print("=" * 60)

print(df.groupby("room_type").apply(
    resumen, include_groups=False
).round(2).to_string())

print("\n" + "=" * 60)
print("2. ERRORES POR ZONA")
print("=" * 60)

print(df.groupby("zona").apply(
    resumen, include_groups=False
).round(2).sort_values("MAE").to_string())

print("\n" + "=" * 60)
print("3. ERRORES POR TRAMOS DE PRECIO REAL")
print("=" * 60)

df["tramo_precio"] = pd.cut(
    df["precio_real"],
    bins=[0, 100, 200, 300, 500, 1000, float("inf")],
    labels=["0-100", "100-200", "200-300",
            "300-500", "500-1000", ">1000"],
    include_lowest=True
)

print(df.groupby("tramo_precio", observed=True).apply(
    resumen, include_groups=False
).round(2).to_string())

print("\n" + "=" * 60)
print("4. LOS 10 MAYORES ERRORES")
print("=" * 60)

columnas = [
    "zona", "room_type", "precio_real",
    "precio_predicho", "error_absoluto"
]

print(df.nlargest(10, "error_absoluto")[columnas].round(2).to_string(index=False))

# Guardar análisis
salida = RAIZ / "reports"
salida.mkdir(exist_ok=True)

df.to_csv(salida / "analisis_errores_test.csv", index=False)

print("\nAnálisis guardado en reports/analisis_errores_test.csv")
