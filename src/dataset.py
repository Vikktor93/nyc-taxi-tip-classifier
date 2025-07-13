import pandas as pd
import os

# Ruta por defecto al dataset local
DEFAULT_DATA_PATH = os.path.join(
    os.getcwd(), 'data', 'raw', 'yellow_tripdata_2020-01.parquet'
)

# Carga y limpieza de datos
def load_data(path: str = None) -> pd.DataFrame:

    # Si no se pasa path, se usa el archivo por defecto
    if path is None:
        path = DEFAULT_DATA_PATH

    # Carga de datos
    df = pd.read_parquet(path)

    # Eliminar registros con nulos en columnas clave
    df = df.dropna(subset=['tip_amount', 'total_amount', 'tpep_pickup_datetime', 'payment_type'])

    # Crear porcentaje de propina
    df['tip_pct'] = df['tip_amount'] / df['total_amount']

    # Variable objetivo: 1 si propina > 20%, sino 0
    df['y'] = (df['tip_pct'] > 0.2).astype(int)

    # Asegurar tipo datetime
    df['pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])

    return df