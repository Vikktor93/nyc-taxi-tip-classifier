import pandas as pd

# Carga y limpieza de datos
def load_data(path: str) -> pd.DataFrame:
    # Carga de datos
    path = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2020-01.parquet'
    df = pd.read_parquet(path)

    # Se elimina los registros con valores nulos en columnas clave
    df = df.dropna(subset=['tip_amount', 'total_amount', 'pickup_datetime', 'payment_type'])

    # Creación del porcentaje de propina
    df['tip_pct'] = df['tip_amount'] / df['total_amount']

    # Se define la variable objetivo: 1 si propina > 20%, sino 0
    df['y'] = (df['tip_pct'] > 0.2).astype(int)

    # Se convierte pickup_datetime a datetime si no lo está
    if not pd.api.types.is_datetime64_any_dtype(df['pickup_datetime']):
        df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

    return df
