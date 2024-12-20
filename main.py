import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine

# Ruta al archivo CSV
file_path = './data/global_temperature.csv'

# Función para cargar, limpiar y llenar valores faltantes
def load_and_clean_data(file_path):
    try:
        # Cargar datos
        data = pd.read_csv(file_path, skiprows=1)

        # Renombrar columnas
        data.columns = [
            "Year", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
            "J-D", "D-N", "DJF", "MAM", "JJA", "SON"
        ]

        # Reemplazar valores "***" por NaN
        data.replace("***", np.nan, inplace=True)

        # Convertir todas las columnas numéricas a tipo float
        cols = data.columns.drop("Year")
        data[cols] = data[cols].apply(pd.to_numeric, errors='coerce')

        # Llenar valores faltantes con promedios
        data.fillna(data.mean(), inplace=True)

        # Mostrar un resumen de los datos limpios
        print(f"Datos después de la limpieza: {data.shape[0]} filas y {data.shape[1]} columnas.")
        return data
    except Exception as e:
        print(f"Error al limpiar los datos: {e}")
        return None

# Función para visualizar tendencias con resaltado
def plot_trend_with_highlights(data):
    threshold = 1.0  # Definir umbral de anomalía significativa
    extreme_years = data[data['J-D'] > threshold]

    plt.figure(figsize=(10, 6))
    plt.plot(data['Year'], data['J-D'], label='Anomalía Anual (J-D)', color='b')
    plt.scatter(extreme_years['Year'], extreme_years['J-D'], color='r', label='Anomalías Significativas')
    plt.xlabel('Año')
    plt.ylabel('Anomalía de Temperatura (°C)')
    plt.title('Tendencia de la Anomalía de Temperatura Global (con Resaltado)')
    plt.legend()
    plt.grid()
    plt.show()

# Función para graficar promedios por década
def plot_decade_means(data):
    data['Decade'] = (data['Year'] // 10) * 10
    decade_means = data.groupby('Decade').mean()

    plt.figure(figsize=(10, 6))
    plt.bar(decade_means.index, decade_means['J-D'], color='orange', label='Promedio Década')
    plt.xlabel('Década')
    plt.ylabel('Anomalía de Temperatura Promedio (°C)')
    plt.title('Promedio de Anomalías por Década')
    plt.legend()
    plt.grid(axis='y')
    plt.show()

    return decade_means

# Función para generar mapa de calor de correlaciones
def plot_correlation_heatmap(data):
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.iloc[:, 1:13].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Mapa de Correlación entre Meses')
    plt.show()

if __name__ == "__main__":
    # Cargar y limpiar los datos
    cleaned_data = load_and_clean_data(file_path)

    # Visualización de tendencias con resaltado
    if cleaned_data is not None:
        plot_trend_with_highlights(cleaned_data)

        # Comparación de promedios por década
        decade_means = plot_decade_means(cleaned_data)

        # Mapa de calor de correlaciones
        plot_correlation_heatmap(cleaned_data)
        
        
from sqlalchemy import create_engine

# Configuración de la base de datos PostgreSQL
DB_USER = 'postgres'  # Cambia esto si tienes otro usuario
DB_PASSWORD = 'postgres'  # Coloca aquí tu contraseña de PostgreSQL
DB_HOST = 'localhost'  # Por defecto, es localhost
DB_PORT = '5432'  # Puerto por defecto de PostgreSQL
DB_NAME = 'climate_project'  # Nombre de tu base de datos

# Función para cargar los datos en PostgreSQL
def load_to_postgresql(data, table_name='temperature_anomalies'):
    try:
        # Crear la conexión a PostgreSQL
        engine = create_engine(f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

        # Cargar los datos en la tabla
        data.to_sql(table_name, engine, if_exists='replace', index=False)
        print(f"Datos cargados exitosamente en PostgreSQL en la tabla '{table_name}'.")
    except Exception as e:
        print(f"Error al cargar los datos en PostgreSQL: {e}")

if __name__ == "__main__":
    # Asegúrate de que los datos estén limpios
    cleaned_data = load_and_clean_data('./data/global_temperature.csv')

    if cleaned_data is not None:
        # Cargar los datos limpios a PostgreSQL
        load_to_postgresql(cleaned_data)

