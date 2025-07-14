<p align="left">
   <img src="https://img.shields.io/badge/Status-En%20Desarrollo-green?style=plastic">
   <img src="https://img.shields.io/badge/Python-3776AB?style=plastic&logo=python&logoColor=white"/>
   <img src="https://img.shields.io/badge/Jupyter-%23e58f1a.svg?style=plastic&logo=Jupyter&logoColor=white"/>

<img src="./assets/banner-nyc.png"/>

## **Tarea 1: Reestructuración y Evaluación de Modelo de Machine Learning**
### **NYC Taxi Tip Classifier**

Repositorio que implementa un pipeline de Ciencia de Datos para clasificar viajes de taxi en NYC según si la propina es alta (>20%) o no.

### Estructura del proyecto

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

### Requisitos

- Python 3.10+  
- VS Code o su editor favorito
- Anaconda o el entorno virtual de su preferencia con las siguientes librerías:
  - pandas  
  - scikit-learn  
  - joblib  
  - matplotlib  

### Instrucciones de instalación

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

### Uso
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
4. Generación de gráficos:
    ```bash
    python -m src.plots reports\metrics.csv --out reports\figures\f1_monthly.png
    ```

### Tabla de Resultados Mensuales

| Mes     | Muestra | F1-score |
| ------- | ------- | -------- |
| 2020-01 | 100 000 | 0.0329   |
| 2020-02 | 100 000 | 0.0293   |
| 2020-03 | 100 000 | 0.0368   |

Para evaluar con datos completos, elimina  ```--sample_size ``` y ajusta la tabla con el número real de registros.


### Conclusiones
1. El F1-score es muy bajo (< 0.05) en los meses evaluados y muestra ligeras variaciones (0.0329 vs. 0.0293), lo que indica que el modelo tiene un desempeño limitado para la clase minoritaria (propinas altas).
2. Algunos factores que puedan explicar esta variación:
    - Existe un desbalance de clases, ya que la proporción de viajes con propinas altas es muy baja (< 5%).
    - Cambio estacional a principios de 2020, la pandemia de COVID-19 provocó restricciones de movilidad y alteró drásticamente los patrones de uso de taxis y propinas. Esto pudo impactar tanto en el volumen de viajes como en el comportamiento de propina en distintos meses.
3. Analizando los resultados se puede recomendar lo siguiente:
    - Para manejar el desbalance se puede utilizar undersampling o emplear SMOTE.
    - Incluir algunos indicadores de restricciones como por ejemplo, niveles de confinamiento o movilidad, para capturar el efecto de la pandemia en el comportamiento de propina.