"""Interfaz para investigación académica de mensajes potencialmente fraudulentos."""
import json
from pathlib import Path
import joblib
import streamlit as st
from predecir import analizar

st.set_page_config(page_title='Detector de mensajes sospechosos | PLN', page_icon='🛡️', layout='centered')
BASE = Path(__file__).resolve().parent
RUTA = BASE / 'resultados' / 'modelo.joblib'
st.title('🛡️ Análisis de mensajes SMS')
st.caption('Clasificador experimental de posibles fraudes. No comprueba remitentes, enlaces ni hechos.')
if not RUTA.exists():
    st.error('Falta resultados/modelo.joblib. Ejecuta el entrenamiento siguiendo README.md.')
    st.stop()

@st.cache_resource
def cargar_modelo(ruta, modificacion):
    # joblib debe usarse solo con modelos propios y confiables.
    return joblib.load(ruta)

try:
    modelo = cargar_modelo(str(RUTA), RUTA.stat().st_mtime_ns)
except Exception as exc:
    st.error('No se pudo cargar el modelo. No reutilices modelos antiguos: vuelve a entrenar con este proyecto.')
    with st.expander('Detalle técnico'):
        st.exception(exc)
    st.stop()

with st.form('analisis'):
    mensaje = st.text_area('Mensaje SMS (sin datos privados)', height=155, max_chars=5000,
                           placeholder='Ej.: Has sido seleccionado para un bono. Confirma tu tarjeta...')
    enviar = st.form_submit_button('Analizar mensaje', type='primary', use_container_width=True)
if enviar:
    if len(mensaje.strip()) < 8:
        st.warning('Escribe al menos 8 caracteres.')
    else:
        try:
            r = analizar(modelo, mensaje)
            if r['clase'] == 1:
                st.warning('Posible fraude: se encontraron patrones similares a ejemplos etiquetados como sospechosos.')
            else:
                st.info('Sin patrones suficientes para la clase sospechosa. NO significa que el mensaje sea seguro.')
            st.write('**Clasificación del modelo:**', r['etiqueta'])
            if 'puntuacion' in r:
                st.metric('Puntuación interna de la clase sospechosa', f"{r['puntuacion']:.1%}")
                st.caption('No es una probabilidad calibrada de estafa real.')
            if 'margen' in r:
                st.metric('Margen hacia clase sospechosa', f"{r['margen']:+.3f}")
                st.caption('El margen no es porcentaje ni probabilidad.')
        except Exception as exc:
            st.error('Error al analizar el mensaje.')
            with st.expander('Detalle técnico'):
                st.exception(exc)

st.divider()
st.subheader('Evaluación del experimento')
met = BASE / 'resultados' / 'metricas.json'
if met.exists():
    try:
        d = json.loads(met.read_text(encoding='utf-8'))
        st.write('**Modelo:**', d.get('modelo_seleccionado', 'Sin dato'))
        st.write('**Mensajes:**', d.get('n_total', 'Sin dato'))
        st.write('**F1 validación cruzada:**', d.get('f1_cv_seleccionado', 'Sin dato'))
        if d.get('origen') == 'demostracion_sintetica':
            st.warning('Modelo DEMOSTRATIVO entrenado con datos sintéticos. Sus métricas no acreditan desempeño ante estafas reales en Perú.')
        with st.expander('Reporte y matriz de confusión'):
            st.json(d)
    except (OSError, ValueError) as exc:
        st.warning(f'No se pudieron leer las métricas: {exc}')
st.info('No compartas DNI, contraseñas ni códigos de verificación. Confirma cualquier oferta por canales oficiales.')
