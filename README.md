# Clasificación experimental de mensajes potencialmente fraudulentos en Perú

**Alcance:** prototipo académico PLN; corpus `datos_demo.csv` es **sintético, pequeño y no representativo**. La clase 1 significa *ejemplo etiquetado como posible fraude*, no fraude confirmado. No usar para decisiones reales de seguridad.

## Instalación en VS Code (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe preparar_datos.py --entrada datos_demo.csv --salida datos_limpios.csv
.\.venv\Scripts\python.exe modelo.py --datos datos_limpios.csv --salida resultados --origen demostracion_sintetica
.\.venv\Scripts\python.exe -m streamlit run app.py
```

**Importante:** no copies `resultados/modelo.joblib` de versiones antiguas. Este proyecto genera uno nuevo sin referencias a `__main__`.

## Probar por consola

```powershell
.\.venv\Scripts\python.exe predecir.py --modelo resultados/modelo.joblib --texto "Has sido seleccionado para un bono. Registra tu tarjeta para cobrarlo."
```

## Evaluación externa

`evaluar.py` requiere un CSV independiente `texto,etiqueta`, con ambas clases y al menos 10 casos únicos por clase. No se incluye un corpus externo validado. Cuando lo tengas:

```powershell
.\.venv\Scripts\python.exe evaluar.py --modelo resultados/modelo.joblib --datos evaluacion_externa.csv
```

No uses mensajes de prueba ni variantes muy parecidas para ajustar el modelo. Deduplica y separa por campaña/remitente cuando sea posible, para reducir fuga de información.

## Metodología

1. Etiquetado explícito: 0=sin evidencia de fraude, 1=posible fraude, con criterios verificables.
2. Validación, normalización y deduplicación de textos.
3. División estratificada 80/20; búsqueda de hiperparámetros por validación cruzada solo en entrenamiento.
4. Comparación TF-IDF (palabras y caracteres) con Regresión Logística, SVM lineal y Naive Bayes.
5. Selección por F1 de clase positiva y reporte de precisión, recall, F1 y matriz de confusión en prueba.
6. Evaluación externa futura en mensajes peruanos reales, anonimizados y con consentimiento/base legal.

## Limitaciones

- Los datos sintéticos reflejan patrones redactados manualmente: las métricas pueden estar infladas.
- El clasificador no valida enlaces, remitentes, instituciones ni intenciones reales.
- Un resultado negativo nunca garantiza seguridad.
- No incluir DNI, números de tarjeta, claves, teléfonos o códigos OTP en datos de entrenamiento.
- No cargar archivos joblib de fuentes no confiables.

## Archivos

`normalizacion.py`, `preparar_datos.py`, `modelo.py`, `predecir.py`, `evaluar.py`, `app.py`, `datos_demo.csv`, `requirements.txt`.
