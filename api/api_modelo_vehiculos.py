"""
API REST para Predicción de Categoría de Vehículos — SRI 2026
==============================================================

Proyecto:
    Práctico Experimental — Minería de Datos
    Universidad Estatal Amazónica (UEA)
    Código: UEA-L-UFPTI-009

Modelo:
    Random Forest Classifier
    n_estimators = 50
    max_depth    = 10

Métricas reportadas durante la evaluación:
    Accuracy test : 94.72%
    CV Mean (k=5): 86.00%

Descripción:
    API REST que predice la categoría de precio de un vehículo:

        Bajo | Medio | Alto

    El sistema recibe nombres legibles de:
        - Marca
        - Clase
        - Tipo de combustible

    Posteriormente realiza la conversión:

        nombre → código → características del modelo
        → StandardScaler → Random Forest → predicción

Endpoints:
    GET  /
    GET  /salud
    GET  /catalogos
    GET  /modelo/info
    POST /predecir
    GET  /docs

======================================================================
EJECUCIÓN LOCAL EN VS CODE
======================================================================

1. Instalar dependencias:

    pip install -r requirements.txt

2. Estructura mínima esperada:

    mineria-sri-2026/
    │
    ├── api/
    │   ├── api_modelo_vehiculos.py
    │   ├── catalogo_vehiculos.py
    │   ├── modelo_vehiculos_sri.pkl
    │   └── scaler_vehiculos_sri.pkl
    │
    └── webapp/
        └── index.html

3. Ejecutar desde la raíz del proyecto:

    uvicorn api.api_modelo_vehiculos:app --reload --port 8000

4. Abrir documentación:

    http://localhost:8000/docs

======================================================================
DEPLOY EN RENDER
======================================================================

Start Command recomendado:

    uvicorn api.api_modelo_vehiculos:app --host 0.0.0.0 --port $PORT

No utilizar --reload en producción.

======================================================================
VARIABLES DE ENTORNO OPCIONALES
======================================================================

ANIO_REFERENCIA
    Año utilizado para calcular la antigüedad.

    Ejemplo:
        ANIO_REFERENCIA=2026

API_CORS_ORIGINS
    Orígenes permitidos separados por coma.

    Ejemplo:
        API_CORS_ORIGINS=http://localhost:5500,https://tu-frontend.onrender.com

======================================================================
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# =============================================================================
# 1. CONFIGURACIÓN GENERAL
# =============================================================================

API_VERSION = "2.1.0"

# Directorio donde está ubicado este archivo.
BASE_DIR = Path(__file__).resolve().parent

# Archivos generados durante el entrenamiento del modelo.
MODELO_PATH = BASE_DIR / "modelo_vehiculos_sri.pkl"
SCALER_PATH = BASE_DIR / "scaler_vehiculos_sri.pkl"

# Año de referencia para calcular la antigüedad.
# Se puede cambiar mediante una variable de entorno.
ANIO_REFERENCIA = int(
    os.getenv("ANIO_REFERENCIA", "2026")
)

# Identificación del modelo utilizado.
MODEL_NAME = "RandomForestClassifier"

MODEL_PARAMS = {
    "n_estimators": 50,
    "max_depth": 10,
}

# Métricas documentadas del proyecto.
#
# Estas métricas corresponden al proceso de evaluación del proyecto
# y no representan una garantía de acierto para cada predicción individual.
MODEL_METRICS = {
    "accuracy_test": 0.9472,
    "cv_mean_k5": 0.8600,
}


# =============================================================================
# 2. CONFIGURACIÓN DE LOGS
# =============================================================================

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format=(
        "%(asctime)s | %(levelname)s | "
        "%(name)s | %(message)s"
    ),
)

logger = logging.getLogger("sri-vehiculos-api")


# =============================================================================
# 3. IMPORTACIÓN DEL CATÁLOGO
# =============================================================================

# Se intenta primero la importación utilizada al ejecutar Uvicorn
# desde la raíz del proyecto:
#
#     uvicorn api.api_modelo_vehiculos:app
#
# Se mantiene un segundo intento para facilitar la ejecución directa
# desde determinadas configuraciones de VS Code.

try:
    from api.catalogo_vehiculos import (
        MARCAS,
        CLASES,
        COMBUSTIBLES,
    )
except ModuleNotFoundError:
    from catalogo_vehiculos import (
        MARCAS,
        CLASES,
        COMBUSTIBLES,
    )


# =============================================================================
# 4. CONFIGURACIÓN CORS
# =============================================================================

def obtener_origenes_cors() -> list[str]:
    """
    Obtiene los orígenes permitidos para consumir la API.

    La configuración recomendada para producción es utilizar
    la variable de entorno API_CORS_ORIGINS.

    Ejemplo:

        API_CORS_ORIGINS=http://localhost:5500,https://frontend.com

    Para desarrollo, si no se define la variable, se permiten
    algunos orígenes locales y el comodín '*'.
    """

    valor = os.getenv("API_CORS_ORIGINS", "").strip()

    if valor:
        return [
            origen.strip()
            for origen in valor.split(",")
            if origen.strip()
        ]

    return [
        "*"
    ]


CORS_ORIGINS = obtener_origenes_cors()


# =============================================================================
# 5. CREACIÓN DE LA APLICACIÓN FASTAPI
# =============================================================================

app = FastAPI(
    title="API Minería de Datos — SRI Vehículos 2026",
    description=(
        "API REST para predecir la categoría de precio de vehículos "
        "utilizando un modelo Random Forest previamente entrenado. "
        "Acepta nombres legibles de marca, clase y combustible."
    ),
    version=API_VERSION,
    contact={
        "name": "Práctico Experimental — Minería de Datos UEA",
        "url": "https://www.uea.edu.ec",
    },
)


# CORS permite que la WebApp JavaScript pueda comunicarse con FastAPI.
#
# allow_credentials=False es intencional cuando se utiliza un origen
# abierto o '*'. No se están manejando cookies ni sesiones de usuario.
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Accept"],
)


# =============================================================================
# 6. CARGA DEL MODELO Y SCALER
# =============================================================================

def cargar_recursos_ml() -> tuple[object, object]:
    """
    Carga el modelo Random Forest y el StandardScaler.

    La función valida primero que ambos archivos existan y captura
    errores de lectura para facilitar el diagnóstico durante el
    despliegue en Render o ejecución local.
    """

    archivos_faltantes: list[str] = []

    if not MODELO_PATH.exists():
        archivos_faltantes.append(str(MODELO_PATH))

    if not SCALER_PATH.exists():
        archivos_faltantes.append(str(SCALER_PATH))

    if archivos_faltantes:
        raise FileNotFoundError(
            "No se encontraron los archivos requeridos: "
            + ", ".join(archivos_faltantes)
        )

    try:
        modelo_cargado = joblib.load(
            MODELO_PATH
        )

        scaler_cargado = joblib.load(
            SCALER_PATH
        )

        logger.info(
            "Modelo cargado correctamente: %s",
            MODELO_PATH,
        )

        logger.info(
            "Scaler cargado correctamente: %s",
            SCALER_PATH,
        )

        return (
            modelo_cargado,
            scaler_cargado,
        )

    except Exception as exc:
        logger.exception(
            "No fue posible cargar los archivos del modelo."
        )

        raise RuntimeError(
            "Error al cargar el modelo o el scaler."
        ) from exc


# Carga de los recursos ML al iniciar la API.
modelo, scaler = cargar_recursos_ml()


# =============================================================================
# 7. ESQUEMAS PYDANTIC
# =============================================================================

CategoriaPrecio = Literal[
    "Bajo",
    "Medio",
    "Alto",
]


class VehiculoInput(BaseModel):
    """
    Esquema de entrada para la predicción.

    La API recibe nombres legibles y realiza internamente la
    conversión a los códigos utilizados durante el entrenamiento.
    """

    marca: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description=(
            "Marca del vehículo. Ejemplo: Toyota"
        ),
        examples=["Toyota"],
    )

    clase: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description=(
            "Clase del vehículo. Ejemplo: Automóvil"
        ),
        examples=["Automóvil"],
    )

    tipo_combustible: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description=(
            "Tipo de combustible. Ejemplo: Gasolina"
        ),
        examples=["Gasolina"],
    )

    cilindraje: int = Field(
        ...,
        ge=1,
        le=10000,
        description=(
            "Cilindraje del vehículo en centímetros cúbicos."
        ),
        examples=[150],
    )

    anio_modelo: int = Field(
        ...,
        ge=1990,
        le=ANIO_REFERENCIA,
        description=(
            "Año del modelo del vehículo."
        ),
        examples=[2024],
    )


class PrediccionOutput(BaseModel):
    """
    Esquema de salida generado por el modelo.
    """

    categoria_precio: CategoriaPrecio

    probabilidad: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Probabilidad asociada a la categoría predicha."
        ),
    )

    # Distribución opcional de probabilidades.
    #
    # Esto permite que la WebApp pueda mostrar:
    #
    # Bajo  : 0.10
    # Medio : 0.25
    # Alto  : 0.65
    #
    # Si el modelo no dispone de predict_proba(), será None.
    probabilidades: dict[str, float] | None = Field(
        default=None,
        description=(
            "Probabilidades calculadas para cada categoría."
        ),
    )

    modelo_usado: str

    marca_ingresada: str

    clase_ingresada: str

    combustible_ingresado: str

    antiguedad_anios: int


# =============================================================================
# 8. FUNCIONES AUXILIARES
# =============================================================================

def normalizar_texto(valor: str) -> str:
    """
    Elimina espacios innecesarios al principio, al final y entre palabras.
    """

    return " ".join(
        str(valor).strip().split()
    )


def obtener_codigo(
    valor: str,
    catalogo: dict[str, int],
    nombre_campo: str,
) -> int:
    """
    Busca el código correspondiente a un nombre del catálogo.

    Primero intenta coincidencia exacta.
    Después permite coincidencia sin distinguir mayúsculas/minúsculas.

    Ejemplos válidos:

        Toyota
        TOYOTA
        toyota
    """

    valor_limpio = normalizar_texto(
        valor
    )

    # Coincidencia exacta.
    if valor_limpio in catalogo:
        return catalogo[valor_limpio]

    # Coincidencia ignorando mayúsculas/minúsculas.
    for nombre, codigo in catalogo.items():

        if nombre.casefold() == valor_limpio.casefold():
            return codigo

    raise HTTPException(
        status_code=400,
        detail=(
            f"{nombre_campo} '{valor}' no reconocido. "
            "Consulte los valores disponibles en /catalogos."
        ),
    )


def calcular_antiguedad(
    anio_modelo: int,
) -> int:
    """
    Calcula la antigüedad del vehículo.

    Ejemplo:

        Año modelo = 2024
        Año referencia = 2026
        Antigüedad = 2
    """

    return max(
        0,
        ANIO_REFERENCIA - anio_modelo,
    )


def normalizar_categoria(
    valor: object,
) -> CategoriaPrecio:
    """
    Convierte la salida del modelo a una categoría válida.

    Se aceptan equivalencias comunes como:

        Bajo / bajo / LOW
        Medio / medio / MEDIUM
        Alto / alto / HIGH
    """

    texto = str(
        valor
    ).strip().casefold()

    equivalencias = {
        "bajo": "Bajo",
        "low": "Bajo",

        "medio": "Medio",
        "medium": "Medio",

        "alto": "Alto",
        "high": "Alto",
    }

    categoria = equivalencias.get(
        texto
    )

    if categoria is None:
        raise RuntimeError(
            "El modelo devolvió una categoría "
            f"no válida: {valor!r}"
        )

    return categoria


def validar_features_modelo(
    dataframe: pd.DataFrame,
) -> None:
    """
    Comprueba que el número de variables enviadas coincida
    con el número de variables esperado por el scaler.

    Esto ayuda a detectar incompatibilidades entre el código de la API
    y los archivos generados durante el entrenamiento.
    """

    numero_features = getattr(
        scaler,
        "n_features_in_",
        None,
    )

    if (
        numero_features is not None
        and numero_features != dataframe.shape[1]
    ):
        raise RuntimeError(
            "Incompatibilidad de variables del modelo: "
            f"el scaler espera {numero_features} features, "
            f"pero la API está enviando {dataframe.shape[1]}."
        )


# =============================================================================
# 9. ENDPOINT RAÍZ
# =============================================================================

@app.get("/")
def root() -> dict:
    """
    Devuelve información general de la API.
    """

    return {
        "api": (
            "Minería de Datos — "
            "SRI Vehículos 2026"
        ),
        "estado": "operativa",
        "version": API_VERSION,
        "modelo": MODEL_NAME,
        "parametros_modelo": MODEL_PARAMS,
        "metricas_evaluacion": MODEL_METRICS,
        "anio_referencia": ANIO_REFERENCIA,

        "endpoints": {
            "docs": "/docs",
            "salud": "/salud",
            "catalogos": "/catalogos",
            "modelo_info": "/modelo/info",
            "predecir": "/predecir",
        },
    }


# =============================================================================
# 10. HEALTH CHECK
# =============================================================================

@app.get("/salud")
def salud() -> dict:
    """
    Endpoint utilizado para comprobar que la API está funcionando
    y que los recursos del modelo fueron cargados correctamente.
    """

    return {
        "status": "ok",
        "modelo_cargado": modelo is not None,
        "scaler_cargado": scaler is not None,
        "modelo": MODEL_NAME,
        "version_api": API_VERSION,
        "anio_referencia": ANIO_REFERENCIA,
    }


# =============================================================================
# 11. INFORMACIÓN DEL MODELO
# =============================================================================

@app.get("/modelo/info")
def modelo_info() -> dict:
    """
    Devuelve información técnica del modelo actualmente cargado.

    Este endpoint es útil para documentación, mantenimiento y
    comprobación del despliegue.
    """

    clases_modelo = getattr(
        modelo,
        "classes_",
        [],
    )

    return {
        "modelo": MODEL_NAME,
        "version_api": API_VERSION,
        "parametros": MODEL_PARAMS,
        "metricas_evaluacion": MODEL_METRICS,
        "anio_referencia": ANIO_REFERENCIA,

        "clases": [
            str(clase)
            for clase in clases_modelo
        ],

        "features": [
            "MARCA_cod",
            "CLASE_cod",
            "TIPO COMBUSTIBLE_cod",
            "CILINDRAJE",
            "antiguedad_anios",
        ],
    }


# =============================================================================
# 12. CATÁLOGOS
# =============================================================================

@app.get("/catalogos")
def obtener_catalogos() -> dict:
    """
    Devuelve las categorías disponibles para los dropdowns
    de la WebApp.

    La respuesta contiene:

        marcas
        clases
        combustibles
    """

    return {
        "marcas": list(
            MARCAS.keys()
        ),

        "clases": list(
            CLASES.keys()
        ),

        "combustibles": list(
            COMBUSTIBLES.keys()
        ),
    }


# =============================================================================
# 13. PREDICCIÓN
# =============================================================================

@app.post(
    "/predecir",
    response_model=PrediccionOutput,
)
def predecir(
    vehiculo: VehiculoInput,
) -> PrediccionOutput:
    """
    Genera una predicción de categoría de precio.

    Flujo de procesamiento:

        1. Recibir datos desde la WebApp.
        2. Normalizar nombres.
        3. Traducir nombres → códigos.
        4. Calcular antigüedad.
        5. Construir DataFrame.
        6. Aplicar StandardScaler.
        7. Ejecutar Random Forest.
        8. Calcular probabilidades.
        9. Retornar resultado estructurado.
    """

    try:

        # =============================================================
        # 13.1 Normalizar entradas
        # =============================================================

        marca = normalizar_texto(
            vehiculo.marca
        )

        clase = normalizar_texto(
            vehiculo.clase
        )

        combustible = normalizar_texto(
            vehiculo.tipo_combustible
        )


        # =============================================================
        # 13.2 Convertir nombres → códigos
        # =============================================================

        marca_cod = obtener_codigo(
            marca,
            MARCAS,
            "Marca",
        )

        clase_cod = obtener_codigo(
            clase,
            CLASES,
            "Clase",
        )

        combustible_cod = obtener_codigo(
            combustible,
            COMBUSTIBLES,
            "Combustible",
        )


        # =============================================================
        # 13.3 Calcular antigüedad
        # =============================================================

        antiguedad = calcular_antiguedad(
            vehiculo.anio_modelo
        )


        # =============================================================
        # 13.4 Construir vector de entrada
        # =============================================================
        #
        # IMPORTANTE:
        # El orden y nombres de estas columnas deben coincidir
        # con el proceso utilizado durante el entrenamiento.
        # =============================================================

        X_nuevo = pd.DataFrame(
            [
                {
                    "MARCA_cod": marca_cod,
                    "CLASE_cod": clase_cod,
                    "TIPO COMBUSTIBLE_cod": combustible_cod,
                    "CILINDRAJE": vehiculo.cilindraje,
                    "antiguedad_anios": antiguedad,
                }
            ]
        )


        # =============================================================
        # 13.5 Validar compatibilidad con el scaler
        # =============================================================

        validar_features_modelo(
            X_nuevo
        )


        # =============================================================
        # 13.6 Aplicar StandardScaler
        # =============================================================

        X_escalado = scaler.transform(
            X_nuevo
        )


        # =============================================================
        # 13.7 Ejecutar predicción
        # =============================================================

        prediccion = modelo.predict(
            X_escalado
        )[0]

        categoria = normalizar_categoria(
            prediccion
        )


        # =============================================================
        # 13.8 Calcular probabilidades
        # =============================================================

        probabilidad_predicha = 0.0

        probabilidades_respuesta: (
            dict[str, float] | None
        ) = None

        if hasattr(
            modelo,
            "predict_proba",
        ):

            probabilidades = (
                modelo.predict_proba(
                    X_escalado
                )[0]
            )

            clases_modelo = list(
                getattr(
                    modelo,
                    "classes_",
                    [],
                )
            )

            probabilidades_respuesta = {}

            for clase_modelo, probabilidad in zip(
                clases_modelo,
                probabilidades,
            ):

                categoria_modelo = normalizar_categoria(
                    clase_modelo
                )

                probabilidades_respuesta[
                    categoria_modelo
                ] = round(
                    float(probabilidad),
                    4,
                )

            probabilidad_predicha = (
                probabilidades_respuesta.get(
                    categoria,
                    0.0,
                )
            )


        # =============================================================
        # 13.9 Registro en logs
        # =============================================================

        logger.info(
            "Predicción | marca=%s | clase=%s | "
            "combustible=%s | cilindraje=%s | "
            "anio=%s | antiguedad=%s | categoria=%s | "
            "probabilidad=%.4f",
            marca,
            clase,
            combustible,
            vehiculo.cilindraje,
            vehiculo.anio_modelo,
            antiguedad,
            categoria,
            probabilidad_predicha,
        )


        # =============================================================
        # 13.10 Retornar resultado
        # =============================================================

        return PrediccionOutput(

            categoria_precio=categoria,

            probabilidad=round(
                probabilidad_predicha,
                4,
            ),

            probabilidades=(
                probabilidades_respuesta
            ),

            modelo_usado=MODEL_NAME,

            marca_ingresada=marca,

            clase_ingresada=clase,

            combustible_ingresado=combustible,

            antiguedad_anios=antiguedad,
        )


    # HTTPException generada por validaciones controladas.
    except HTTPException:
        raise


    # Cualquier otro error se registra en el servidor.
    # No se expone el traceback al usuario.
    except Exception as exc:

        logger.exception(
            "Error durante la predicción."
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "No fue posible generar la predicción. "
                "Revise los logs de la API para obtener "
                "el detalle técnico."
            ),
        ) from exc


# =============================================================================
# 14. EJECUCIÓN DIRECTA
# =============================================================================

if __name__ == "__main__":

    import uvicorn

    # Esta sección está pensada principalmente para desarrollo local.
    # En Render se recomienda utilizar el Start Command configurado
    # en el servicio:

    # uvicorn api.api_modelo_vehiculos:app --host 0.0.0.0 --port $PORT

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(
            os.getenv("PORT", "8000")
        ),
    )