"""Entrenamiento reproducible; selección por CV SOLO dentro de entrenamiento."""
import argparse
import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from normalizacion import normalizar_serie
from preparar_datos import cargar

def construir():
    # El preprocesamiento se aplica dentro del pipeline también en producción.
    return Pipeline([
        ('normalizar', FunctionTransformer(normalizar_serie, validate=False)),
        ('tfidf', TfidfVectorizer(sublinear_tf=True)),
        ('clasificador', LogisticRegression(max_iter=2000))
    ])

def entrenar(datos, salida, origen):
    df = cargar(datos)
    X_ent, X_pru, y_ent, y_pru = train_test_split(
        df.texto, df.etiqueta, test_size=0.2, stratify=df.etiqueta, random_state=42)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    configuraciones = [
        {'clasificador': [LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)], 'clasificador__C': [0.5, 2]},
        {'clasificador': [LinearSVC(class_weight='balanced', random_state=42)], 'clasificador__C': [0.5, 2]},
        {'clasificador': [MultinomialNB()], 'clasificador__alpha': [0.5, 1.0]},
    ]
    grids = []
    for cfg in configuraciones:
        for analizador, ngramas in [('word', (1, 2)), ('char_wb', (3, 5))]:
            grids.append({'tfidf__analyzer': [analizador], 'tfidf__ngram_range': [ngramas], **cfg})
    busqueda = GridSearchCV(construir(), grids, scoring='f1', cv=cv, n_jobs=-1, refit=True)
    busqueda.fit(X_ent, y_ent)
    pred = busqueda.predict(X_pru)
    salida = Path(salida)
    salida.mkdir(parents=True, exist_ok=True)
    joblib.dump(busqueda.best_estimator_, salida / 'modelo.joblib')
    reporte = classification_report(y_pru, pred, labels=[0, 1], output_dict=True, zero_division=0)
    metricas = {
        'origen': origen, 'n_total': len(df), 'n_entrenamiento': len(X_ent),
        'n_prueba': len(X_pru), 'f1_cv_seleccionado': round(float(busqueda.best_score_), 4),
        'modelo_seleccionado': type(busqueda.best_estimator_.named_steps['clasificador']).__name__,
        'parametros': {k: str(v) for k, v in busqueda.best_params_.items()},
        'reporte_prueba': reporte,
        'matriz_confusion': confusion_matrix(y_pru, pred, labels=[0, 1]).tolist(),
        'advertencia': 'Si origen=demostracion_sintetica, estas metricas NO prueban deteccion de fraude real.'
    }
    (salida / 'metricas.json').write_text(json.dumps(metricas, indent=2, ensure_ascii=False), encoding='utf-8')
    pd.DataFrame({'texto': X_pru.to_numpy(), 'real': y_pru.to_numpy(), 'predicho': pred}).to_csv(salida / 'predicciones_prueba.csv', index=False, encoding='utf-8-sig')
    print(json.dumps(metricas, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--datos', default='datos_limpios.csv')
    p.add_argument('--salida', default='resultados')
    p.add_argument('--origen', default='demostracion_sintetica')
    a = p.parse_args()
    entrenar(a.datos, a.salida, a.origen)
