# main.py
import streamlit as st
import pandas as pd
from models import load_all_models
from preprocess import preprocess_data
from utils import show_results, validate_columns, load_data
import numpy as np
import matplotlib.pyplot as plt

# Configuración de la página
st.set_page_config(
    page_title="Predicción de Alteraciones del Crecimiento Fetal",
    layout="wide",
    initial_sidebar_state="expanded")

# Título y descripción
with st.container():
    st.title("Predicción de Alteraciones del Crecimiento Fetal")
    st.markdown(
        "**Herramienta basada en Machine Learning para predecir alteraciones en el crecimiento fetal en ratas Wistar.**"
    )

# Cargar modelos
try:
    models = load_all_models()
    main_model = models['main']
    imputer = models['imputer']
    scaler = models['scaler']
    rfr_models = models['rfr']
except Exception as e:
    st.error(f"Error cargando modelos: {e}")
    st.stop()

# Selección del método de entrada en la sidebar
with st.sidebar:
    st.header("Configuración y Parámetros")
    input_method = st.radio("Seleccionar modo de entrada:",
                            ["📝 Manual", "📂 Archivo"])

    st.markdown("---")
    st.subheader("Visualización de Resultados")
    chart_type = st.selectbox("Tipo de gráfico de distribución",
                              ["Barra", "Pie"])

    st.markdown("---")
    st.subheader("Filtro de Probabilidad")
    threshold = st.slider(
        "Umbral de probabilidad para mostrar predicciones (%)",
        min_value=0,
        max_value=100,
        value=0,
        step=5)

    st.markdown(
        "*(Solo se mostrarán predicciones con probabilidad mayor al umbral seleccionado)*"
    )

with st.container():
    input_data = None

    if input_method == "📝 Manual":
        st.subheader("Entrada Manual de Datos")
        col1, col2 = st.columns(2)
        with col1:
            grupo = st.number_input("Grupo", min_value=0, max_value=3, step=1)
            glicdia14 = st.number_input("Glucosa día 14",
                                        min_value=0.0,
                                        step=0.1)
            glicdia20 = st.number_input("Glucosa día 20",
                                        min_value=0.0,
                                        step=0.1)
            creat = st.number_input("Creatinina", min_value=0.0, step=0.1)
        with col2:
            col = st.number_input("Colesterol", min_value=0.0, step=0.1)
            trig = st.number_input("Triglicéridos", min_value=0.0, step=0.1)
            vldl = st.number_input("VLDL", min_value=0.0, step=0.1)
            insul = st.number_input("Insulina", min_value=0.0, step=0.1)
            hemglic = st.number_input("Hemoglobina Glicosilada",
                                      min_value=0.0,
                                      step=0.1)

        input_data = pd.DataFrame([[
            grupo, glicdia14, glicdia20, creat, col, trig, vldl, insul, hemglic
        ]],
                                  columns=[
                                      'Grupo', 'glicdia14', 'glicdia20',
                                      'Creat', 'Col', 'Trig', 'VLDL', 'Insul',
                                      'hemglic'
                                  ])

        # Validar que no hay campos vacíos o en cero
        if (input_data.isnull().any().any()) or (input_data == 0).any().any():
            st.warning(
                "⚠️ Todos los campos deben estar completos y mayores a cero.")
            input_data = None  # Evita que continúe a predicción

    else:
        st.subheader("Carga de Archivo")
        uploaded_file = st.file_uploader("Subir archivo (CSV o Excel)",
                                         type=["csv", "xlsx"])
        if uploaded_file:
            file_type = 'csv' if uploaded_file.name.endswith(
                '.csv') else 'xlsx'
            input_data = load_data(uploaded_file, file_type)
            if input_data is not None:
                required_cols = [
                    'Grupo', 'glicdia14', 'glicdia20', 'Creat', 'Col', 'Trig',
                    'VLDL', 'Insul', 'hemglic'
                ]
                if validate_columns(input_data, required_cols):
                    st.dataframe(input_data,
                                 height=150,
                                 use_container_width=True)
    
    
# Botón de predicción
with st.container():
    if st.button("Predecir") and input_data is not None:
        try:
            with st.spinner("🔍 Analizando datos. Por favor espere..."):
                processed_data = preprocess_data(input_data, imputer, scaler,
                                                 rfr_models)
                if processed_data is not None:
                    predicted_classes = main_model.predict(processed_data)
                    probabilities = main_model.predict_proba(processed_data)
                    # Filtro interactivo: sólo se muestran predicciones con probabilidad mayor al umbral
                    mask = np.max(probabilities, axis=1) * 100 >= threshold
                    if not mask.any():
                        st.warning(
                            "Ninguna predicción supera el umbral establecido.")
                    else:
                        # Se filtran tanto las clases como las probabilidades
                        filtered_classes = predicted_classes[mask]
                        filtered_probs = probabilities[mask]
                        # Mostrar resultados con función reutilizable
                        show_results(filtered_classes, filtered_probs)

                        # Ejemplo de visualización interactiva basada en la selección del gráfico
                        st.markdown("---")
                        st.subheader("Visualización Interactiva")
                        # Datos para gráfico de distribución
                        dist_data = pd.DataFrame({
                            'Categoría': ['PEG', 'AEG', 'GEG'],
                            'Casos': [
                                np.sum(filtered_classes == 1),
                                np.sum(filtered_classes == 2),
                                np.sum(filtered_classes == 3)
                            ]
                        })
                        if chart_type == "Barra":
                            st.bar_chart(dist_data.set_index('Categoría'))
                        else:
                            # Gráfico de pastel (pie chart) utilizando matplotlib
                            fig, ax = plt.subplots()
                            ax.pie(dist_data['Casos'],
                                   labels=dist_data['Categoría'],
                                   autopct='%1.1f%%',
                                   startangle=90)
                            ax.axis('equal')
                            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error en la predicción: {e}")
