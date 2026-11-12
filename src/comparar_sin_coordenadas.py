
from pathlib import Path
import joblib
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import KFold, cross_validate

RAIZ = Path(__file__).resolve().parent.parent

# Cargar entrenamiento (NO utilizamos test)
X = pd.read_csv(RAIZ / "data/processed/X_train.csv")
y = pd.read_csv(RAIZ / "data/processed/y_train.csv")["price"]

# Modelo definitivo
modelo = joblib.load(
    RAIZ / "models/mejor_modelo.joblib"
)

# Crear dos modelos independientes sin entrenar
modelo_A = clone(modelo)
modelo_B = clone(modelo)

# Modelo B: eliminar latitud y longitud
X_sin_coords = X.drop(
    columns=["latitude", "longitude"]
)

# Acceder al preprocesador del modelo B
preprocesador = modelo_B.regressor.named_steps["preprocessor"]

# Actualizar las columnas numéricas
nuevos_transformadores = []

for nombre, transformador, columnas in preprocesador.transformers:

    if nombre == "num":
        columnas = [
            columna for columna in columnas
            if columna not in ["latitude", "longitude"]
        ]

    nuevos_transformadores.append(
        (nombre, transformador, columnas)
    )

preprocesador.transformers = nuevos_transformadores

# Misma validación cruzada para ambos modelos
cv = KFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

def evaluar(nombre, estimador, datos):

    print(f"\nEvaluando {nombre}...", flush=True)

    resultados = cross_validate(
        estimador,
        datos,
        y,
        cv=cv,
        scoring={
            "MAE": "neg_mean_absolute_error",
            "RMSE": "neg_root_mean_squared_error",
            "R2": "r2"
        },
        n_jobs=1
    )

    mae = -resultados["test_MAE"].mean()
    rmse = -resultados["test_RMSE"].mean()
    r2 = resultados["test_R2"].mean()

    print(f"MAE:  {mae:.2f} €")
    print(f"RMSE: {rmse:.2f} €")
    print(f"R²:   {r2:.4f}")

    return mae, rmse, r2


print("=" * 50)
print("COMPARACIÓN RANDOM FOREST - SMARTSTAY AI")
print("=" * 50)

resultado_A = evaluar(
    "Modelo A - 11 variables",
    modelo_A,
    X
)

resultado_B = evaluar(
    "Modelo B - 9 variables",
    modelo_B,
    X_sin_coords
)

print("\n" + "=" * 50)
print("COMPARACIÓN FINAL")
print("=" * 50)

print(f"MAE modelo A: {resultado_A[0]:.2f} €")
print(f"MAE modelo B: {resultado_B[0]:.2f} €")

diferencia = resultado_B[0] - resultado_A[0]

print(f"Diferencia MAE (B - A): {diferencia:+.2f} €")

if diferencia < 0:
    print("El modelo sin coordenadas obtiene menor MAE.")
elif diferencia > 0:
    print("El modelo con coordenadas obtiene menor MAE.")
else:
    print("Ambos modelos obtienen el mismo MAE.")
