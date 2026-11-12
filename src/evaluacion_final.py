
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# Rutas del proyecto
RAIZ = Path(__file__).resolve().parent.parent
DATOS = RAIZ / "data" / "processed"
RESULTADOS = RAIZ / "reports"
RESULTADOS.mkdir(exist_ok=True)

# Cargar el modelo definitivo
modelo = joblib.load(RAIZ / "models" / "mejor_modelo.joblib")

# Cargar test reservado
X_test = pd.read_csv(DATOS / "X_test.csv")
y_test = pd.read_csv(DATOS / "y_test.csv")["price"]

# Verificaciones
assert len(X_test) == len(y_test), "Tamaños diferentes"
assert X_test.columns.tolist() == list(modelo.feature_names_in_), (
    "Las variables no coinciden con el modelo"
)

# Evaluación final: una sola predicción sobre test
y_pred = modelo.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nEVALUACIÓN FINAL - SMARTSTAY AI")
print("=" * 40)
print(f"Registros de test: {len(y_test)}")
print(f"MAE:  {mae:.2f} €")
print(f"RMSE: {rmse:.2f} €")
print(f"R²:   {r2:.4f}")

# Guardar métricas para Streamlit
pd.DataFrame([{
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2
}]).to_csv(RESULTADOS / "metricas_test.csv", index=False)

# Gráfico: precio real frente a predicho
plt.figure(figsize=(9, 6))
plt.scatter(y_test, y_pred, alpha=0.25, s=8)

limite = max(float(y_test.max()), float(np.max(y_pred)))
plt.plot([0, limite], [0, limite], linestyle="--", color="red")

plt.xlabel("Precio real (€)")
plt.ylabel("Precio predicho (€)")
plt.title("SmartStay AI - Evaluación final Random Forest")
plt.grid(alpha=0.2)
plt.tight_layout()

plt.savefig(
    RESULTADOS / "real_vs_predicho.png",
    dpi=150
)
plt.close()

print("\nArchivos guardados en reports/")
print("- metricas_test.csv")
print("- real_vs_predicho.png")
