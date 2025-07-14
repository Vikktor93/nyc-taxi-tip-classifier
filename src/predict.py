import os
import sys
import argparse
import joblib
from src.dataset import load_data
from src.build_features import create_features
from sklearn.metrics import f1_score, classification_report

# Se comprueba que la carpeta raíz del proyecto esté en PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Función que carga un modelo serializado y evalúa sobre el dataset
def evaluate_model(model_path: str,
                   data_path: str,
                   sample_size: int = None):
    # Carga de modelo
    clf = joblib.load(model_path)
    
    # Carga y limpiar datos
    df = load_data(data_path)

    # Muestreo si se especifica
    if sample_size is not None:
        df = df.head(sample_size)
        print(f"Usando muestra de {sample_size} filas para evaluación.")

    # Generación de features y etiqueta
    X = create_features(df)
    y = df['y']

    # Predicción y métricas
    preds = clf.predict(X)
    print(classification_report(y, preds, digits=4))
    f1 = f1_score(y, preds)
    print(f"F1-score total: {f1:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evalúa un modelo de propinas de taxi'
    )
    parser.add_argument(
        'model_path',
        help='Ruta al modelo .joblib'
    )
    parser.add_argument(
        'data_path',
        help='Ruta o URL al Parquet de evaluación'
    )
    parser.add_argument(
        '--sample_size',
        type=int,
        default=None,
        help='Número de filas a muestrear para evaluación rápida'
    )
    args = parser.parse_args()
    evaluate_model(
        args.model_path,
        args.data_path,
        sample_size=args.sample_size
    )