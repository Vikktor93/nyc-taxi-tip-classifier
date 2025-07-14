<p align="left">
   <img src="https://img.shields.io/badge/Status-En%20Desarrollo-green?style=plastic">
   <img src="https://img.shields.io/badge/Python-3776AB?style=plastic&logo=python&logoColor=white"/>
   <img src="https://img.shields.io/badge/Jupyter-%23e58f1a.svg?style=plastic&logo=Jupyter&logoColor=white"/>

<img src="./assets/banner-nyc.png"/>

## **Tarea 1: Reestructuración y Evaluación de Modelo de Machine Learning**

## NYC Taxi Tip Classifier

Repositorio que implementa un pipeline de Ciencia de Datos para clasificar viajes de taxi en NYC según si la propina es alta (>20%) o no.

## Estructura del proyecto

```
Tarea-1/
├── data/
│ └── raw/
│ ├── yellow_tripdata_2020-01.parquet
│ └── yellow_tripdata_2020-02.parquet
├── models/
│ └── random_forest_taxi_tip.joblib
├── reports/
│ ├── metrics.csv
│ └── figures/
├── src/
│ ├── init.py
│ ├── dataset.py
│ ├── build_features.py
│ ├── train.py
│ ├── predict.py
│ ├── evaluate_monthly.py
│ └── plots.py
└── README.md

```

## Requisitos

- Python 3.10+  
- Conda o entorno virtual con las librerías:
  - pandas  
  - scikit-learn  
  - joblib  
  - matplotlib  

## Instrucciones de instalación

1. Clona este repositorio:  
   ```bash
   git clone https://github.com/Vikktor93/nyc-taxi-tip-classifier
   cd Tarea-1
   ```bash

2. Crea y activa un entorno Conda:
    ```bash
    conda create -n taxi-tip python=3.10.13
    conda activate taxi-tip
    pip install -r requirements.txt
    ```

## Uso
1. Entrenamiento:
    ```bash
    python -m src.train data\raw\yellow_tripdata_2020-01.parquet models\random_forest_taxi_tip.joblib --sample_size 100000 --test_size 0.2
    ```
2. Evaluación mes a mes:
    ```bash
    python -m src.predict models\random_forest_taxi_tip.joblib data\raw\yellow_tripdata_2020-02.parquet --sample_size 100000
    ```

3. Evaluación mensual automatizada:
    ```bash
    python -m src.evaluate_monthly models\random_forest_taxi_tip.joblib data\raw reports\metrics.csv --sample_size 100000
    ```
4. Generación de gráfica:
    ```bash
    python -m src.plots reports\metrics.csv --out reports\figures\f1_monthly.png
    ```

## Tabla de Resultados Mensuales

| Mes     | Cantidad de ejemplos | F1-score |
| ------- | -------------------- | -------- |
| 2020-01 | 100 000 (muestra)    | 0.0329   |
| 2020-02 | 100 000 (muestra)    | 0.0293   |
| 2020-03 | 100 000 (muestra)    | 0.0368   |

Para evaluar con datos completos, elimina  ```--sample_size ``` y ajusta la tabla con el número real de registros.


