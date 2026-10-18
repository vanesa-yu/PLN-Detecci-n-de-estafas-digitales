"""Transformaciones deterministas, compatibles con joblib (sin __main__)."""
import re
import unicodedata

def normalizar(texto):
    texto = unicodedata.normalize('NFKC', str(texto)).lower().strip()
    texto = re.sub(r'https?://\S+|www\.\S+', ' marcadorurl ', texto)
    texto = re.sub(r'\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b', ' marcadorcorreo ', texto)
    texto = re.sub(r'(?<!\w)(?:s/\s*|us\$\s*|\$\s*)\d[\d,.]*', ' marcadormonto ', texto)
    texto = re.sub(r'\b\+?\d[\d\s()-]{7,}\d\b', ' marcadortelefono ', texto)
    texto = re.sub(r'\b\d+\b', ' marcadornumero ', texto)
    return re.sub(r'\s+', ' ', texto).strip()

def normalizar_serie(textos):
    return [normalizar(t) for t in textos]
