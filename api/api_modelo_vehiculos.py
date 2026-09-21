"""
API REST para Predicción de Categoría de Vehículos — SRI 2026
=================================================================
Asignatura: Minería de Datos — UEA-L-UFPTI-009
Modelo: Random Forest (n_estimators=50, max_depth=10)
Accuracy test: 94.72% | CV_Mean (k=5): 86.00%

----------------------------------------------------------------------------- 
¿CÓMO EJECUTAR EN VS CODE?
-----------------------------------------------------------------------------
1) Instalar dependencias:
   pip install -r requirements.txt

2) Asegurarse de tener en la misma carpeta:
   - modelo_vehiculos_sri.pkl   (descargado de Colab)
   - scaler_vehiculos_sri.pkl   (descargado de Colab)
   - api_modelo_vehiculos.py    (este archivo)

3) Levantar el servidor:
   uvicorn api_modelo_vehiculos:app --reload --port 8000

4) Probar:
   GET  http://localhost:8000/
   GET  http://localhost:8000/salud
   POST http://localhost:8000/predecir
   Docs http://localhost:8000/docs
=============================================================================
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal
import joblib
import os
import pandas as pd


# =============================================================================
# 1. Configuración de la aplicación
# =============================================================================
app = FastAPI(
    title="API Minería de Datos — SRI Vehículos 2026",
    description=(
        "API REST que predice la categoría de precio de un vehículo "
        "(Bajo / Medio / Alto) basándose en sus características técnicas. "
        "Modelo: Random Forest (n_estimators=50, max_depth=10)."
    ),
    version="1.0.0",
    contact={
        "name": "Práctico Experimental — Minería de Datos UEA",
        "url": "https://www.uea.edu.ec",
    },
)

# CORS — Permite que la WebApp (Live Server en :5500) consuma la API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # En producción, especifica tu dominio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# 2. Carga del modelo y escalador entrenados
# =============================================================================
MODELO_PATH = os.path.join(os.path.dirname(__file__), "modelo_vehiculos_sri.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "scaler_vehiculos_sri.pkl")

if not os.path.exists(MODELO_PATH) or not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(
        f"No se encontraron los archivos .pkl en {os.path.dirname(__file__)}. "
        "Descárgalos desde el notebook de Colab (panel izquierdo → Archivos → "
        "clic derecho → Descargar)."
    )

modelo = joblib.load(MODELO_PATH)   # RandomForestClassifier entrenado
scaler = joblib.load(SCALER_PATH)   # StandardScaler ajustado a X_train


# =============================================================================
# 3. Esquemas de entrada y salida (Pydantic DTOs)
# =============================================================================
class VehiculoInput(BaseModel):
    """
    Esquema de entrada para la predicción.
    Todos los valores numéricos deben estar ya codificados (LabelEncoder)
    y en las mismas unidades del dataset original.
    """
    marca_cod: int = Field(
        ..., ge=0, le=300,
        description="Código numérico de la MARCA (LabelEncoder)"
    )
    clase_cod: int = Field(
        ..., ge=0, le=50,
        description="Código numérico de la CLASE (LabelEncoder)"
    )
    tipo_combustible_cod: int = Field(
        ..., ge=0, le=10,
        description="Código de TIPO COMBUSTIBLE (LabelEncoder)"
    )
    cilindraje: int = Field(
        ..., ge=0, le=10000,
        description="Cilindraje del vehículo en cc"
    )
    anio_modelo: int = Field(
        ..., ge=1990, le=2026,
        description="Año del modelo del vehículo"
    )


class PrediccionOutput(BaseModel):
    """
    Esquema de salida de la predicción.
    """
    categoria_precio: Literal["Bajo", "Medio", "Alto"]
    probabilidad: float
    modelo_usado: str


# =============================================================================
# 4. Endpoints
# =============================================================================
@app.get("/")
def root():
    """Endpoint raíz: información general de la API."""
    return {
        "api": "Minería de Datos — SRI Vehículos 2026",
        "modelo": "RandomForestClassifier (n_estimators=50, max_depth=10)",
        "exactitud_test": 0.9472,
        "cv_mean_k5": 0.8600,
        "endpoints": {
            "docs": "/docs",
            "predecir": "/predecir (POST)",
            "salud": "/salud"
        }
    }


@app.get("/salud")
def salud():
    """Health-check del servicio."""
    return {"status": "ok", "modelo_cargado": True}


@app.post("/predecir", response_model=PrediccionOutput)
def predecir(v: VehiculoInput):
    """
    Predice la categoría de precio de un vehículo.

    Algoritmo:
      1) Construye un DataFrame con las 5 características codificadas
      2) Calcula antiguedad_anios = 2026 - anio_modelo
      3) Aplica StandardScaler (Z-score) ya ajustado
      4) Ejecuta el modelo .predict() y .predict_proba()
      5) Retorna la categoría y la probabilidad asociada
    """
    try:
        # Feature engineering: derivar antigüedad
        antiguedad = max(0, 2026 - v.anio_modelo)

        # Construir vector de entrada en el orden exacto
        X_nuevo = pd.DataFrame([{
            'MARCA_cod':                v.marca_cod,
            'CLASE_cod':                v.clase_cod,
            'TIPO COMBUSTIBLE_cod':     v.tipo_combustible_cod,
            'CILINDRAJE':               v.cilindraje,
            'antiguedad_anios':         antiguedad
        }])

        # Normalización Z-Score:  z = (x - mu) / sigma
        X_escalado = scaler.transform(X_nuevo)

        # Predicción
        categoria = modelo.predict(X_escalado)[0]

        # Probabilidad de la clase predicha
        probabilidades = modelo.predict_proba(X_escalado)[0]
        clases = modelo.classes_
        idx = list(clases).index(categoria)
        prob = float(probabilidades[idx])

        return PrediccionOutput(
            categoria_precio=categoria,
            probabilidad=round(prob, 4),
            modelo_usado="RandomForestClassifier"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error en la predicción: {str(e)}"
        )


# =============================================================================
# 5. EJECUCIÓN DIRECTA (python api_modelo_vehiculos.py)
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api_modelo_vehiculos:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )