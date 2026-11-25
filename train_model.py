import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# 1. Cargar datos
df = pd.read_csv("clientes_historicos.csv")
print("Filas y columnas:", df.shape)
print("Distribución del objetivo:\n", df["regresa_30_dias"].value_counts(normalize=True).round(3))

# 2. Separar variables de entrada (X) y objetivo (y). cliente_id no se usa para entrenar.
y = df["regresa_30_dias"]
X = df.drop(columns=["regresa_30_dias", "cliente_id"])

categoricas = ["canal_compra", "producto_favorito"]
numericas = [c for c in X.columns if c not in categoricas]
print("Categorías de canal_compra:", sorted(X["canal_compra"].unique()))
print("Categorías de producto_favorito:", sorted(X["producto_favorito"].unique()))

# 3. Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Preprocesamiento + modelo en un solo pipeline
preprocesamiento = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categoricas),
    ("num", "passthrough", numericas),
])
modelo = Pipeline([
    ("prep", preprocesamiento),
    ("rf", RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")),
])
modelo.fit(X_train, y_train)

# 5. Evaluar en el conjunto de prueba
pred = modelo.predict(X_test)
print("\nReporte de clasificación:\n", classification_report(y_test, pred))
print("Matriz de confusión:\n", confusion_matrix(y_test, pred))
print("ROC-AUC:", round(roc_auc_score(y_test, modelo.predict_proba(X_test)[:, 1]), 3))

# 6. Guardar el pipeline completo (preprocesamiento + modelo)
joblib.dump(modelo, "model.joblib")
print("\nModelo guardado en model.joblib")
