
from pathlib import Path
import joblib
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import KFold, cross_validate

RAIZ = Path(__file__).resolve().parent.parent

# Cargar datos de entrenamiento
X = pd.read_csv(RAIZ / "data/processed/X_train.csv")
y = pd.read_csv(RAIZ / "data/processed/y_train.csv")["price"]

# Cargar modelo original
modelo = joblib.load(
    RAIZ / "models/mejor_modelo.joblib"
)

# Dos copias independientes del modelo
modelo_A = clone(modelo)
modelo_C = clone(modelo)

# Eliminar estancia_larga de los datos
X_sin_estancia = X.drop(columns=["estancia_larga"])

# Acceder al preprocesador del modelo C
preprocesador = modelo_C.regressor.named_steps["preprocessor"]

# Modificar las columnas categóricas
nuevos_transformadores = []

for nombre, transformador, columnas in preprocesador.transformers:

    if nombre == "cat":
        columnas = [
            columna for columna in columnas
            if columna != "estancia_larga"
        ]

    nuevos_transformadores.append(
        (nombre, transformador, columnas)
    )

preprocesador.transformers = nuevos_transformadores

# Misma validación cruzada que el experimento anterior
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


print("=" * 55)
print("SMARTSTAY AI - COMPARACIÓN DE VARIABLES")
print("=" * 55)

resultado_A = evaluar(
    "Modelo A - 11 variables",
    modelo_A,
    X
)

resultado_C = evaluar(
    "Modelo C - 10 variables (sin estancia_larga)",
    modelo_C,
    X_sin_estancia
)

print("\n" + "=" * 55)
print("RESULTADOS FINALES")
print("=" * 55)

print(f"MAE modelo A: {resultado_A[0]:.2f} €")
print(f"MAE modelo C: {resultado_C[0]:.2f} €")

diferencia = resultado_C[0] - resultado_A[0]

print(f"Diferencia MAE (C - A): {diferencia:+.2f} €")

if diferencia < 0:
    print("MEJORA: eliminar estancia_larga reduce el error.")
elif diferencia > 0:
    print("EMPEORA: conviene mantener estancia_larga.")
else:
    print("EMPATE: ambos modelos obtienen el mismo MAE.")
