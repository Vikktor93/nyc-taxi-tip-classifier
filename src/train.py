# File: src/train.py
import os
import sys
import argparse
import joblib
from src.dataset import load_data
from src.build_features import create_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# Se comprueba que la carpeta raíz del proyecto esté en PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Se entrena un RandomForest para clasificación de propinas con opción de muestreo
def train_model(data_path: str,
                model_path: str,
                test_size: float = 0.2,
                sample_size: int = None):
    
    # Carga y limpieza
    df = load_data(data_path)

    # Se toma una muestra reducida
    if sample_size is not None:
        df = df.head(sample_size)
        print(f"Usando muestra de {sample_size} filas para entrenamiento")

    # Generación de features y etiqueta
    X = create_features(df)
    y = df['y']

    # División train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )

    # Definición y entrenamiento del modelo (paralelizado)
    clf = RandomForestClassifier(
        n_estimators=100,
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    clf.fit(X_train, y_train)

    # Evaluación en test
    preds = clf.predict(X_test)
    f1 = f1_score(y_test, preds)
    print(f"F1-score en test: {f1:.4f}")

    # Serialización del model
    joblib.dump(clf, model_path)
    print(f"Modelo guardado en: {model_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Entrena clasificador de propinas con RandomForest'
    )
    parser.add_argument(
        'data_path',
        help='Ruta o URL al archivo Parquet de entrada'
    )
    parser.add_argument(
        'model_path',
        help='Ruta para guardar el modelo (.joblib)'
    )
    parser.add_argument(
        '--test_size',
        type=float,
        default=0.2,
        help='Fracción del dataset para test (por defecto 0.2)'
    )
    parser.add_argument(
        '--sample_size',
        type=int,
        default=None,
        help='Número de filas a muestrear (df.head) para pruebas rápidas'
    )
    args = parser.parse_args()
    train_model(
        args.data_path,
        args.model_path,
        test_size=args.test_size,
        sample_size=args.sample_size
    )

