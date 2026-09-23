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
   - catalogo_vehiculos.py      (catálogo de mapeo nombre → código)
   - api_modelo_vehiculos.py    (este archivo)

3) Levantar el servidor:
   uvicorn api_modelo_vehiculos:app --reload --port 8000

4) Probar:
   GET  http://localhost:8000/
   GET  http://localhost:8000/catalogos       ← listas para los dropdowns
   GET  http://localhost:8000/salud
   POST http://localhost:8000/predecir         ← acepta NOMBRES legibles
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

# Importar catálogo de mapeo nombre → código
from api.catalogo_vehiculos import MARCAS, CLASES, COMBUSTIBLES


# =============================================================================
# 1. Configuración de la aplicación
# =============================================================================
app = FastAPI(
    title="API Minería de Datos — SRI Vehículos 2026",
    description=(
        "API REST que predice la categoría de precio de un vehículo "
        "(Bajo / Medio / Alto) basándose en sus características técnicas. "
        "Modelo: Random Forest (n_estimators=50, max_depth=10). "
        "Acepta NOMBRES legibles (marca, clase, combustible) en lugar de "
        "códigos numéricos, facilitando el uso desde la WebApp."
    ),
    version="2.0.0",
    contact={
        "name": "Práctico Experimental — Minería de Datos UEA",
        "url": "https://www.uea.edu.ec",
    },
)

# CORS — Permite que la WebApp (Live Server en :5500) consuma la API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

modelo = joblib.load(MODELO_PATH)
scaler = joblib.load(SCALER_PATH)


# =============================================================================
# 3. Esquemas de entrada y salida (Pydantic DTOs)
# =============================================================================
class VehiculoInput(BaseModel):
    """
    Esquema de entrada para la predicción.
    Acepta NOMBRES legibles en lugar de códigos numéricos.
    La API traduce los nombres internamente.
    """
    marca: str = Field(
        ..., description="Nombre de la marca (ej: 'Shineray', 'Chevrolet')"
    )
    clase: str = Field(
        ..., description="Nombre de la clase (ej: 'Motocicleta', 'Automóvil')"
    )
    tipo_combustible: str = Field(
        ..., description="Tipo de combustible (ej: 'Gasolina', 'Diésel')"
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
    """Esquema de salida de la predicción."""
    categoria_precio: Literal["Bajo", "Medio", "Alto"]
    probabilidad: float
    modelo_usado: str
    marca_ingresada: str
    clase_ingresada: str
    combustible_ingresado: str
    antiguedad_anios: int


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
        "version": "2.0.0 — Acepta nombres legibles en lugar de códigos",
        "endpoints": {
            "docs": "/docs",
            "catalogos": "/catalogos",
            "predecir": "/predecir (POST — con nombres)",
            "salud": "/salud"
        }
    }


@app.get("/salud")
def salud():
    """Health-check del servicio."""
    return {"status": "ok", "modelo_cargado": True}


@app.get("/catalogos")
def obtener_catalogos():
    """
    Devuelve las listas de marcas, clases y combustibles disponibles.
    La WebApp las usa para llenar los menús desplegables (dropdowns).
    """
    return {
        "marcas":       list(MARCAS.keys()),
        "clases":       list(CLASES.keys()),
        "combustibles": list(COMBUSTIBLES.keys())
    }


@app.post("/predecir", response_model=PrediccionOutput)
def predecir(v: VehiculoInput):
    """
    Predice la categoría de precio de un vehículo.

    Flujo:
      1) Recibe NOMBRES legibles (marca, clase, tipo_combustible)
      2) Traduce nombres → códigos usando el catálogo
      3) Calcula antiguedad_anios = 2026 - anio_modelo
      4) Aplica StandardScaler (Z-score)
      5) Ejecuta .predict() y .predict_proba()
      6) Retorna la categoría y probabilidad
    """
    try:
        # === TRADUCIR NOMBRES → CÓDIGOS ===
        if v.marca not in MARCAS:
            raise HTTPException(
                status_code=400,
                detail=f"Marca '{v.marca}' no reconocida. "
                       f"Disponibles: {list(MARCAS.keys())}"
            )
        if v.clase not in CLASES:
            raise HTTPException(
                status_code=400,
                detail=f"Clase '{v.clase}' no reconocida. "
                       f"Disponibles: {list(CLASES.keys())}"
            )
        if v.tipo_combustible not in COMBUSTIBLES:
            raise HTTPException(
                status_code=400,
                detail=f"Combustible '{v.tipo_combustible}' no reconocido. "
                       f"Disponibles: {list(COMBUSTIBLES.keys())}"
            )

        marca_cod            = MARCAS[v.marca]
        clase_cod            = CLASES[v.clase]
        tipo_combustible_cod = COMBUSTIBLES[v.tipo_combustible]
        antiguedad           = max(0, 2026 - v.anio_modelo)

        # Construir vector de entrada en el orden exacto del modelo
        X_nuevo = pd.DataFrame([{
            'MARCA_cod':                marca_cod,
            'CLASE_cod':                clase_cod,
            'TIPO COMBUSTIBLE_cod':     tipo_combustible_cod,
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
            modelo_usado="RandomForestClassifier",
            marca_ingresada=v.marca,
            clase_ingresada=v.clase,
            combustible_ingresado=v.tipo_combustible,
            antiguedad_anios=antiguedad
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error en la predicción: {str(e)}"
        )


# =============================================================================
# 5. EJECUCIÓN DIRECTA
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api_modelo_vehiculos:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )