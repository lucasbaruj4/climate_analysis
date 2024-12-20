import streamlit as st
import pandas as pd
import psycopg2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor


# Configuración de la conexión a PostgreSQL
DB_USER = 'postgres'
DB_PASSWORD = 'postgres'
DB_HOST = 'localhost'
DB_PORT = '5432'
DB_NAME = 'climate_project'

# Conectar a PostgreSQL
def connect_to_db():
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        return conn
    except Exception as e:
        st.error(f"Error al conectar a la base de datos: {e}")
        return None

# Consultar datos de la base
def run_query(query):
    conn = connect_to_db()
    if conn:
        try:
            df = pd.read_sql(query, conn)
            conn.close()
            return df
        except Exception as e:
            st.error(f"Error al ejecutar la consulta: {e}")
            conn.close()
            return None
    else:
        return None

# Configurar el dashboard
st.title("Dashboard de Análisis Climático")
st.sidebar.title("Opciones de Visualización")

# Opciones de consulta
options = st.sidebar.selectbox(
    "Selecciona el Insight",
    [
        "Promedios por Década",
        "Años con Anomalías Extremas",
        "Tendencia por Estación",
        "Cambios Año a Año",
        "Top 5 Años Críticos",
        "Predicciones de Machine Learning",
        "Predicciones con Regresión Polinómica",
        "Predicciones con Random Forest",
        "Gráfico Avanzado de Anomalías"
    ]
)
# Función para preparar los datos y entrenar el modelo
def train_and_predict(data):
    X = data[["Year"]]
    y = data["J-D"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    st.write(f"Error cuadrático medio (MSE): {mse:.4f}")
    future_years = pd.DataFrame({"Year": range(2025, 2051)})
    future_predictions = model.predict(future_years)
    future_years["Predicted Anomaly"] = future_predictions
    st.line_chart(future_years.set_index("Year"))

# Función para entrenar un modelo de regresión polinómica
def train_polynomial_regression(data):
    X = data[["Year"]]
    y = data["J-D"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    poly_model = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
    poly_model.fit(X_train, y_train)
    y_pred = poly_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    st.write(f"Error cuadrático medio (MSE - Regresión Polinómica): {mse:.4f}")
    future_years = pd.DataFrame({"Year": range(2025, 2051)})
    future_predictions = poly_model.predict(future_years)
    future_years["Predicted Anomaly"] = future_predictions
    st.line_chart(future_years.set_index("Year"))
    
#Funcion para entrenar el modelo que utiliza Random Forest
def train_random_forest(data):
    # Crear nuevas características
    data["Year_Relative"] = data["Year"] - data["Year"].min()
    
    # Preparar los datos
    X = data[["Year_Relative"]]
    y = data["J-D"]
    
    # Dividir en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Crear el modelo con normalización
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    
    # Evaluar el modelo
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    st.write(f"Error cuadrático medio (MSE - Random Forest): {mse:.4f}")
    
    # Predicciones futuras
    future_years = pd.DataFrame({
        "Year": range(2025, 2051),
        "Year_Relative": range(2025 - data["Year"].min(), 2051 - data["Year"].min())
    })
    future_predictions = pipeline.predict(future_years[["Year_Relative"]])
    
    # Mostrar predicciones
    future_years["Predicted Anomaly"] = future_predictions
    st.subheader("Predicciones Futuras (Random Forest)")
    st.line_chart(future_years.set_index("Year"))


#Funcion para crear el grafico avanzado
def plot_advanced_chart(data):
    fig = px.line(
        data,
        x="Year",
        y="J-D",
        title="Anomalías de Temperatura por Año",
        labels={"J-D": "Anomalía (°C)", "Year": "Año"}
    )
    st.plotly_chart(fig)

if options == "Gráfico Avanzado de Anomalías":
    query = '''
        SELECT "Year", "J-D"
        FROM temperature_anomalies
        ORDER BY "Year";
    '''
    data = run_query(query)
    if data is not None:
        st.subheader("Gráfico Avanzado de Anomalías")
        plot_advanced_chart(data)
    else:
        st.error("No se pudieron cargar los datos.")
# Ejecutar consultas según la opción seleccionada
if options == "Promedios por Década":
    query = '''
        SELECT FLOOR("Year" / 10) * 10 AS "Decade", AVG("J-D") AS "Avg_Anomaly"
        FROM temperature_anomalies
        GROUP BY "Decade"
        ORDER BY "Decade";
    '''
    data = run_query(query)
    if data is not None:
        st.subheader("Promedios de Anomalía por Década")
        st.bar_chart(data.set_index("Decade"))

elif options == "Años con Anomalías Extremas":
    query = '''
        SELECT "Year", "J-D"
        FROM temperature_anomalies
        WHERE "J-D" > 1.0
        ORDER BY "J-D" DESC;
    '''
    data = run_query(query)
    if data is not None:
        st.subheader("Años con Anomalías Extrema")
        st.dataframe(data)

elif options == "Tendencia por Estación":
    query = '''
        SELECT AVG("MAM") AS Avg_MAM, AVG("JJA") AS Avg_JJA, AVG("SON") AS Avg_SON, AVG("DJF") AS Avg_DJF
        FROM temperature_anomalies;
    '''
    data = run_query(query)
    if data is not None:
        st.subheader("Promedio de Anomalías por Estación")
        st.bar_chart(data.T)

elif options == "Cambios Año a Año":
    query = '''
        SELECT "Year", "J-D", "J-D" - LAG("J-D") OVER (ORDER BY "Year") AS "Change"
        FROM temperature_anomalies
        ORDER BY "Year";
    '''
    data = run_query(query)
    if data is not None:
        st.subheader("Cambios de Anomalías Año a Año")
        st.line_chart(data.set_index("Year")["Change"])

elif options == "Top 5 Años Críticos":
    query = '''
        SELECT "Year", "J-D"
        FROM temperature_anomalies
        ORDER BY "J-D" DESC
        LIMIT 5;
    '''
    data = run_query(query)
    if data is not None:
        st.subheader("Top 5 Años Críticos")
        st.table(data)

elif options == "Predicciones de Machine Learning":
    query = '''
        SELECT "Year", "J-D"
        FROM temperature_anomalies
        ORDER BY "Year";
    '''
    data = run_query(query)
    if data is not None:
        st.subheader("Modelo Predictivo")
        train_and_predict(data)
    else:
        st.error("No se pudo cargar los datos para el modelo.")

elif options == "Predicciones con Regresión Polinómica":
    query = '''
        SELECT "Year", "J-D"
        FROM temperature_anomalies
        ORDER BY "Year";
    '''
    data = run_query(query)
    if data is not None:
        st.subheader("Modelo de Regresión Polinómica")
        train_polynomial_regression(data)
    else:
        st.error("No se pudieron cargar los datos para el modelo.")

elif options == "Predicciones con Random Forest":
    query = '''
        SELECT "Year", "J-D"
        FROM temperature_anomalies
        ORDER BY "Year";
    '''
    data = run_query(query)
    if data is not None:
        st.subheader("Modelo de Random Forest")
        train_random_forest(data)
    else:
        st.error("No se pudieron cargar los datos para el modelo.")