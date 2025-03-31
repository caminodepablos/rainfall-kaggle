import streamlit as st
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# Para ejecutar el modelo en Streamlit
# streamlit run app-rainfall-model.py

# -------------------------------------------------------
# CONFIGURACIÓN PÁGINA
# -------------------------------------------------------


# Configuración
st.set_page_config(
    page_title="Rainfall Classifier", 
    layout="wide",
    page_icon = '⛈️',
    initial_sidebar_state = 'expanded')

# Usar una imagen de fondo con CSS y agregar la descripción dentro de la imagen
st.markdown(
    """
    <style>
    :root {
        --primary-color: # 2a7abf;
        --text-color: #414141;
        }
    body, [data-testid="stAppViewContainer"] {
        background-color: white !important;
        color: black !important;
    }
    [data-testid="stSidebar"] {
        background-color: #f0f2f6 !important;
    }
    .container {
        background-image: url('https://images.pexels.com/photos/1529360/pexels-photo-1529360.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1');
        background-size: cover;
        background-position: center;
        text-align: center;
        color: white;
        padding: 200px 0;
    }
    .title {
        font-size: 70px;
        font-weight: bold;
    }
    .description {
        font-size: 18px;
        margin-top: 20px;
    }

    .stButton>button {
        background-color: #12274B;  /* Fondo botón */
        color: white;               /* Color texto */
        border: none;               /* Sin borde */
        padding: 15px 32px;         /* Relleno del botón */
        text-align: center;         /* Alineación del texto */
        text-decoration: none;      /* Sin subrayado */
        display: inline-block;      /* En línea */
        font-size: 16px;            /* Tamaño de fuente */
        margin: 4px 2px;            /* Márgenes */
        cursor: pointer;           /* Cursor en forma de mano */
        border-radius: 12px;        /* Bordes redondeados */
    }
    
    .stButton>button:hover {
        background-color: #1468B1;  /* Fondo al pasar el ratón */
        color: white;
    }

    .stSuccess>div {
        color: #12274B; 
        font-size: 24px;
        font-weight: bold;
    }

    .stError>div {
        color: #e74c3c;  /* Rojo para el error */
        font-size: 18px;
        font-weight: bold;
    }

    .header {
            font-size: 28px;
            font-weight: bold;
            text-align: center;
            color: #4CAF50;
            padding: 10px;
            background-color: #f4f4f4;
            border-radius: 10px;
        }
        
        .slider-container label {
            font-size: 16px;
            font-weight: bold;
            color: #333;
        }
        
        .stSlider {
            background-color: #ececec;
            border-radius: 5px;
            padding: 8px;
        }
    
    </style>
    
    <div class="container">
        <div class="title">⛆ Rainfall Classifier ⛆</div>
        <div class="description">
            Binary Prediction with a Rainfall Dataset - Kaggle Playground Series<br>
            See more 🔗 <a href="https://www.kaggle.com/c/playground-series-s5e3" target="_blank" style="color: white;">https://www.kaggle.com/c/playground-series-s5e3</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------------
# MODELO Y DATASET
# -------------------------------------------------------

modelo = joblib.load("rainfall_model.pkl")
data = pd.read_csv('train.csv')
X = data.drop(columns = ['rainfall'])
y = data['rainfall']
features = X.columns.tolist()

# -------------------------------------------------------
# MENU
# -------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "🧮 All Features Prediction",
    "📊 Data Analysis",
    "✅ Prediction History",
    "🧪 Scenario Comparison"
])

# --------------------------
# TAB 1: Predicción
# --------------------------


with tab1:
    st.header("🧮 All Features Prediction")

    inputs = []
    cols = st.columns(3)
    for i in range(len(features)):
        col = cols[i % 3]
        with col:
            # Cambiar de data[:, i] a data.iloc[:, i] para acceder correctamente a las columnas
            min_val = float(np.min(data.iloc[:, i]))
            max_val = float(np.max(data.iloc[:, i]))
            mean_val = float(np.mean(data.iloc[:, i]))
            if min_val == max_val:
                max_val += 1.0
            val = st.slider(
                label=features[i],
                min_value=min_val,
                max_value=max_val,
                value=mean_val,
                step=0.01
            )
            inputs.append(val)

    if st.button("Predict 🌧️"):
        try:
            pred = modelo.predict([inputs])[0]
            st.success("Result: **Rains** 🌧️" if pred == 1 else "Result: **Doesn't Rain** 🌞 ")
        except Exception as e:
            st.error(f"❌ Error en la predicción: {e}")

# --------------------------
# TAB 2: Exploración
# --------------------------

with tab2:
    st.header("📊 Exploratory Data Analysis")

    # Filtro por lluvia
    st.subheader("🔎 Dataset Filter")
    opcion_clase = st.selectbox("Rain Filter:", ["All", "Rain", "Not Rain"])
    if opcion_clase == "Rain":
        data = data[data["rainfall"] == 1]
    elif opcion_clase == 'Not Rain':
        data = data[data["rainfall"] == 0]
    st.dataframe(data.head(10))

    # Selector de variable
    st.subheader("📈 Univariante Analysis")
    feature_seleccionada = st.selectbox("Select one feature:", features)

    # Histograma interactivo con línea del paciente
    with st.expander("📊 Distribution (Plotly)"):
        fig = px.histogram(
            data,
            x=feature_seleccionada,
            nbins=30,
            color_discrete_sequence=["#7FDBFF"]
        )
        if 'inputs' in locals():
            idx = list(features).index(feature_seleccionada)
            fig.add_vline(
                x=inputs[idx],
                line_dash="dash",
                line_color="crimson",
                annotation_text="Valor actual",
                annotation_position="top right"
            )
        fig.update_layout(
            height=400,
            title=feature_seleccionada,
            template="simple_white"
        )
        st.plotly_chart(fig, use_container_width=True, key="1")

    # Heatmap de correlaciones
    with st.expander("📌 Mapa de calor de correlaciones"):
        corr = data[features].corr().round(2)
        fig = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            zmin=-1, 
            zmax=1,
            aspect="auto"
        )
        fig.update_layout(
            title="Correlaciones entre variables",
            height=600
        )
        st.plotly_chart(fig, use_container_width=True, key="2" )

# --------------------------
# TAB 3: Historial
# --------------------------

with tab3:
    st.header("🕘 Prediction History")

    if "historial" not in st.session_state:
        st.session_state.historial = []

    # Solo se guarda si hay predicción actual disponible
    try:
        input_array = np.array(inputs).reshape(1, -1)
        pred = modelo.predict(input_array)[0]
        proba = modelo.predict_proba(input_array)[0][1]
        
        if st.button("Save Scenario ⬆"):
            st.session_state.historial.append({
                "Prediction": "Rain" if pred == 1 else "Not Rain",
                "Rain Probability": round(proba, 4),
                "Date & Time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            st.success("Scenario Saved ✔️")

    except Exception as e:
        st.info("Create a prediction first.")

    # Mostrar historial
    if st.session_state.historial:
        st.dataframe(pd.DataFrame(st.session_state.historial))

        # Botón para limpiar
        if st.button("🧹 Clean History"):
            st.session_state.historial = []
            st.experimental_rerun()
    else:
        st.info("You don't have any prediction yet.")

# --------------------------
# TAB 4: Comparador
# --------------------------

with tab4:
    st.header("🧪 Scenario Comparison (Top 4 Features)")

    # Obtener las 4 variables más importantes del modelo
    importancias = {
    "feature": ["cloud", "sunshine", "dewpoint", "maxtemp", "mintemp", "id", "pressure", "temparature", "day", "humidity"],
    "importance": [0.765343, 0.109744, 0.056805, 0.017858, 0.013257, 0.012217, 0.011641, 0.007658, 0.005478, 0.0]}

    df_importancia = pd.DataFrame(importancias, index=[8, 9, 6, 3, 5, 0, 2, 4, 1, 7])
    top_features = df_importancia.sort_values("importance", ascending=False)["feature"].head(4).tolist()

    # Mostrar sliders para las top 4
    st.markdown("### 🅰️ Scenery (Original)")
    inputs_a_mod = []
    cols_a = st.columns(2)
    for i, feature in enumerate(top_features):
        idx = list(features).index(feature)
        with cols_a[i % 2]:
            val = st.slider(
                f"{feature} (A)",
                min_value=float(np.min(data.iloc[:, idx])),
                max_value=float(np.max(data.iloc[:, idx])),
                value=float(inputs[idx]),
                step=0.1
            )
            inputs_a_mod.append((idx, val))

    inputs_a = inputs.copy()
    for idx, val in inputs_a_mod:
        inputs_a[idx] = val

    st.markdown("---")

    st.markdown("### 🅱️ Scenery (Modified)")
    inputs_b_mod = []
    cols_b = st.columns(2)
    for i, feature in enumerate(top_features):
        idx = list(features).index(feature)
        with cols_b[i % 2]:
            val = st.slider(
                f"{feature} (B)",
                min_value=float(np.min(data.iloc[:, idx])),
                max_value=float(np.max(data.iloc[:, idx])),
                value=float(inputs[idx]),
                step=0.1
            )
            inputs_b_mod.append((idx, val))

    inputs_b = inputs.copy()
    for idx, val in inputs_b_mod:
        inputs_b[idx] = val

    # Predicciones
    pred_a = modelo.predict([inputs_a])[0]
    proba_a = modelo.predict_proba([inputs_a])[0][1]

    pred_b = modelo.predict([inputs_b])[0]
    proba_b = modelo.predict_proba([inputs_b])[0][1]

    # Resultados comparativos
    st.markdown("### 📊 Results Comparision")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("🅰️ Prediction", "Rain" if pred_a == 1 else "Not Rain")
        st.metric("🅰️ Probability", f"{proba_a:.2%}")

    with col2:
        st.metric("🅱️ Prediction", "Rain" if pred_a == 1 else "Not Rain")
        st.metric("🅱️ Probability", f"{proba_b:.2%}")
