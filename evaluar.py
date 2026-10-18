"""Evaluación externa: no usar el mismo corpus de entrenamiento."""
import argparse
import json
import joblib
from sklearn.metrics import classification_report, confusion_matrix
from preparar_datos import cargar

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--modelo', default='resultados/modelo.joblib')
    p.add_argument('--datos', required=True, help='CSV externo, nunca usado para entrenar ni ajustar')
    a = p.parse_args()
    df = cargar(a.datos)
    modelo = joblib.load(a.modelo)
    pred = modelo.predict(df.texto)
    print(classification_report(df.etiqueta, pred, labels=[0, 1], zero_division=0))
    print('Matriz [0,1]:', json.dumps(confusion_matrix(df.etiqueta, pred, labels=[0, 1]).tolist()))
    errores = df.loc[df.etiqueta.to_numpy() != pred, ['texto', 'etiqueta']].copy()
    errores['predicho'] = pred[df.etiqueta.to_numpy() != pred]
    errores.to_csv('errores_evaluacion.csv', index=False, encoding='utf-8-sig')
    print('Errores guardados en errores_evaluacion.csv')

if __name__ == '__main__':
    main()
