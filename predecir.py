"""Predicción sin asumir que LinearSVC tiene predict_proba."""
import argparse
import joblib

def analizar(modelo, texto):
    clase = int(modelo.predict([texto])[0])
    respuesta = {'clase': clase, 'etiqueta': 'Posible fraude' if clase else 'Sin patrones suficientes'}
    if hasattr(modelo, 'predict_proba'):
        clases = list(modelo.classes_)
        if 1 in clases:
            respuesta['puntuacion'] = float(modelo.predict_proba([texto])[0][clases.index(1)])
    elif hasattr(modelo, 'decision_function'):
        valor = float(modelo.decision_function([texto])[0])
        if list(modelo.classes_)[-1] != 1:
            valor = -valor
        respuesta['margen'] = valor
    return respuesta

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--modelo', default='resultados/modelo.joblib')
    p.add_argument('--texto', required=True)
    a = p.parse_args()
    print(analizar(joblib.load(a.modelo), a.texto))
