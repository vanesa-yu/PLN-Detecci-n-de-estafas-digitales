"""Validación de corpus. Etiqueta 1=posible fraude, 0=no fraude."""
import argparse
from pathlib import Path
import pandas as pd
from normalizacion import normalizar

def cargar(ruta):
    df = pd.read_csv(ruta, encoding='utf-8-sig', dtype={'texto': 'string'})
    requeridas = {'texto', 'etiqueta'}
    if not requeridas.issubset(df.columns):
        raise ValueError('El CSV requiere columnas texto,etiqueta')
    df = df.dropna(subset=['texto', 'etiqueta']).copy()
    df['etiqueta'] = pd.to_numeric(df['etiqueta'], errors='raise').astype(int)
    if not set(df.etiqueta.unique()).issubset({0, 1}):
        raise ValueError('Etiquetas permitidas: 0=no fraude; 1=posible fraude')
    df['texto'] = df.texto.astype(str).str.strip()
    df = df[df.texto.str.len().between(8, 5000)].copy()
    df['clave'] = df.texto.map(normalizar)
    contradicciones = df.groupby('clave').etiqueta.nunique()
    if (contradicciones > 1).any():
        raise ValueError('Hay textos iguales con etiquetas contradictorias')
    df = df.drop_duplicates('clave').drop(columns='clave').reset_index(drop=True)
    if df.etiqueta.nunique() != 2 or df.etiqueta.value_counts().min() < 10:
        raise ValueError('Se necesitan las dos clases y al menos 10 mensajes únicos por clase')
    return df

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--entrada', default='datos_demo.csv')
    p.add_argument('--salida', default='datos_limpios.csv')
    a = p.parse_args()
    df = cargar(a.entrada)
    Path(a.salida).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.salida, index=False, encoding='utf-8-sig')
    print('Preparados:', len(df), 'Distribución:', df.etiqueta.value_counts().to_dict())
