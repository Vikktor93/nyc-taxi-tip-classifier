import os
import sys
import glob
import argparse
import pandas as pd
from joblib import load
from src.dataset import load_data
from src.build_features import create_features
from sklearn.metrics import f1_score

# Se comprueba que la carpeta raíz del proyecto esté en PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Automatiza la evaluación mensual de un modelo entrenado
# Requiere un modelo .joblib y archivos Parquet por mes en una carpeta
def monthly_evaluation(model_path: str,
                       data_dir: str,
                       output_csv: str,
                       sample_size: int = None):
    
    clf = load(model_path)
    results = []
    for file in sorted(glob.glob(os.path.join(data_dir, '*.parquet'))):
        mes = os.path.basename(file).split('_')[-1].split('.')[0]
        df = load_data(file)
        if sample_size:
            df = df.head(sample_size)
        X = create_features(df)
        y = df['y']
        preds = clf.predict(X)
        f1 = f1_score(y, preds)
        results.append({'mes': mes, 'n': len(df), 'f1': f1})
        print(f"Mes {mes}: {len(df)} filas, F1={f1:.4f}")
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"CSV de métricas guardado en: {output_csv}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluación mensual automática')
    parser.add_argument('model_path', help='Ruta al modelo .joblib')
    parser.add_argument('data_dir', help='Carpeta con archivos .parquet por mes')
    parser.add_argument('output_csv', help='Ruta de salida metrics.csv')
    parser.add_argument('--sample_size', type=int, default=None, help='Filas a muestrear para cada mes')
    args = parser.parse_args()
    monthly_evaluation(args.model_path, args.data_dir, args.output_csv, sample_size=args.sample_size)