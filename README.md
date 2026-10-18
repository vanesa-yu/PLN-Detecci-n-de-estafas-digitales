# Clasificación experimental de mensajes SMS potencialmente fraudulentos en el Perú

Prototipo académico de procesamiento de lenguaje natural (PLN) del curso Procesamiento de Lenguaje Natural, Escuela Profesional de Ingeniería de Sistemas, Universidad Nacional del Altiplano (2026-II).

**Alcance.** El corpus `datos_demo.csv` es **sintético, pequeño (60 mensajes) y no representativo**. La clase 1 significa *ejemplo etiquetado como posible fraude*, no fraude confirmado. Las métricas describen únicamente este corpus sintético. **No usar para decisiones reales de seguridad.**

## Contenido del proyecto

| Archivo | Función |
|---|---|
| `normalizacion.py` | Normalización de texto (Unicode NFKC, minúsculas y marcadores de URL, correo, monto, teléfono y número) |
| `preparar_datos.py` | Validación del CSV, deduplicación y generación de `datos_limpios.csv` |
| `modelo.py` | Pipeline TF-IDF + clasificador, búsqueda de hiperparámetros y exportación del modelo |
| `predecir.py` | Clasificación de un mensaje por consola |
| `evaluar.py` | Evaluación sobre un CSV externo independiente |
| `app.py` | Interfaz web en Streamlit |
| `analisis_adicional.py` | Comparación de configuraciones, coeficientes, intervalos de confianza, sensibilidad a la partición, sondeo y figuras del informe |
| `datos_demo.csv` | Corpus sintético (30 mensajes por clase) |
| `requirements.txt` | Dependencias |

## Instalación (VS Code, PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Entrenamiento y aplicación

```powershell
.\.venv\Scripts\python.exe preparar_datos.py --entrada datos_demo.csv --salida datos_limpios.csv
.\.venv\Scripts\python.exe modelo.py --datos datos_limpios.csv --salida resultados --origen demostracion_sintetica
.\.venv\Scripts\python.exe -m streamlit run app.py
```

`modelo.py` crea la carpeta `resultados/` con `modelo.joblib`, `metricas.json` y `predicciones_prueba.csv`.

**Importante:** no copies `modelo.joblib` de versiones antiguas. Este proyecto genera uno nuevo sin referencias a `__main__`. Carga archivos joblib solo si provienen de una fuente confiable.

## Prueba por consola

```powershell
.\.venv\Scripts\python.exe predecir.py --modelo resultados/modelo.joblib --texto "Has sido seleccionado para un bono. Registra tu tarjeta para cobrarlo."
```

La salida incluye la clase, una puntuación interna (Regresión Logística) o un margen (LinearSVC). Ninguno de los dos es una probabilidad calibrada de estafa real.

## Análisis adicional

Requiere haber ejecutado antes `modelo.py` e instalar dos bibliotecas que no figuran en `requirements.txt`:

```powershell
.\.venv\Scripts\python.exe -m pip install matplotlib scipy
.\.venv\Scripts\python.exe analisis_adicional.py
```

Genera `cv_resultados.csv`, `coef.json`, `out/datos.json` y las figuras `out/fig_cv.png`, `out/fig_cm.png` y `out/fig_coef.png`, que se usan en el informe.

## Metodología

1. Etiquetado explícito: 0 = sin evidencia de fraude, 1 = posible fraude.
2. Validación, normalización y deduplicación de textos (longitud entre 8 y 5000 caracteres, sin etiquetas contradictorias, mínimo 10 mensajes únicos por clase).
3. División estratificada 80/20 con semilla 42 (48 mensajes de entrenamiento y 12 de prueba).
4. Búsqueda de hiperparámetros con `GridSearchCV` (3 pliegues, solo sobre entrenamiento): 12 configuraciones, combinando TF-IDF de palabras (1,2) o caracteres (3,5) con Regresión Logística, LinearSVC y Naive Bayes.
5. Selección por F1 de la clase positiva y reporte en prueba de precisión, recall, F1 y matriz de confusión.
6. Evaluación externa futura con mensajes peruanos reales, anonimizados y con base legal.

## Resultados del experimento de demostración

| Elemento | Valor |
|---|---|
| Modelo seleccionado | Regresión Logística, C = 0,5, TF-IDF de palabras (1,2) |
| F1 en validación cruzada (entrenamiento) | 0,9213 (empata con LinearSVC de palabras) |
| Exactitud en prueba (12 mensajes) | 75,00 % (IC 95 %: 42,8 % a 94,5 %) |
| Precisión, recall y F1 de la clase 1 | 71,43 %, 83,33 % y 76,92 % |
| Matriz de confusión [0,1] | [[4, 2], [1, 5]] |

Con 12 mensajes de prueba la incertidumbre es muy alta: al repetir la partición con 20 semillas distintas, la exactitud osciló entre 66,7 % y 100 %.

## Evaluación externa

`evaluar.py` requiere un CSV independiente con columnas `texto,etiqueta`, ambas clases y al menos 10 casos únicos por clase. No se incluye un corpus externo validado.

```powershell
.\.venv\Scripts\python.exe evaluar.py --modelo resultados/modelo.joblib --datos evaluacion_externa.csv
```

Genera `errores_evaluacion.csv` con los mensajes mal clasificados. No uses mensajes de prueba ni variantes muy parecidas para ajustar el modelo. Deduplica y separa por campaña o remitente para reducir la fuga de información.

## Limitaciones

- Los datos sintéticos reflejan patrones redactados manualmente; las métricas pueden estar infladas.
- El modelo se apoya en parte en palabras funcionales («un», «tu», «la») que distinguen el estilo de las dos clases del corpus, no solo en señales de fraude.
- Ante mensajes nuevos las puntuaciones quedan cerca del umbral de decisión; los mensajes legítimos de seguridad bancaria pueden marcarse como sospechosos.
- El clasificador no valida enlaces, remitentes, instituciones ni intenciones reales.
- Un resultado negativo nunca garantiza seguridad.

## Privacidad y ética

- No incluir DNI, números de tarjeta, claves, teléfonos ni códigos OTP en los datos de entrenamiento.
- El tratamiento de mensajes reales debe cumplir la Ley N.° 29733, Ley de Protección de Datos Personales.
- No automatizar bloqueos o sanciones a partir de una predicción aislada.
- No compartir DNI, contraseñas ni códigos de verificación; confirmar cualquier oferta por canales oficiales.

## Verificación de reproducibilidad

El entrenamiento reproduce exactamente los valores de `entrenamiento.log` con Python 3.12.3, scikit-learn 1.8.0 y pandas 3.0.2.
