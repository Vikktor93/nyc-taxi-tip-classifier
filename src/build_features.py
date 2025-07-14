import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Función que genera variables predictoras a partir del DataFrame limpio
def create_features(df: pd.DataFrame) -> pd.DataFrame:

    # Se copia el DataFrame para no modificar el original
    data = df.copy()

    # Características temporales
    data['hour'] = data['pickup_datetime'].dt.hour
    data['day_of_week'] = data['pickup_datetime'].dt.dayofweek

    # Distancia: se usa trip_distance 
    if 'trip_distance' in data.columns:
        data['distance'] = data['trip_distance']

    # One-hot encoding de payment_type
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    ohe = encoder.fit_transform(data[['payment_type']])
    ohe_cols = encoder.get_feature_names_out(['payment_type'])
    df_ohe = pd.DataFrame(ohe, columns=ohe_cols, index=data.index)

    # Selección de columnas finales
    feature_cols = ['hour', 'day_of_week'] + (['distance'] if 'distance' in data else [])
    features = pd.concat([data[feature_cols], df_ohe], axis=1)

    return features
