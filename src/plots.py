import pandas as pd
import matplotlib.pyplot as plt
import argparse

# Función para graficar las métricas mensuales de F1-score
# Toma un archivo CSV con las métricas y genera una gráfica de líneas.
def plot_metrics(csv_path: str, output_path: str = None):
    df = pd.read_csv(csv_path)
    plt.figure()
    plt.plot(df['mes'], df['f1'], marker='o')
    plt.xlabel('Mes')
    plt.ylabel('F1-score')
    plt.title('Variación mensual de F1-score')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"Gráfica guardada en: {output_path}")
    else:
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Graficar métricas mensuales de F1-score')
    parser.add_argument('csv_path', help='Ruta al archivo metrics.csv')
    parser.add_argument('--out', help='Ruta para guardar la figura', default=None)
    args = parser.parse_args()
    plot_metrics(args.csv_path, args.out)
