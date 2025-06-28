import streamlit as st
from models import load_all_models
from preprocess import process_data
from utils import show_results, validate_input_data, load_data, get_manual_input, render_distribution_chart, show_file_instructions
from retrain_page import render_retrain_page
import numpy as np
import os
from pathlib import Path

# Configuración de la página
st.set_page_config(
    page_title="FetalGrowth AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_models():
    """Carga los modelos una sola vez y los mantiene en caché"""
    try:
        return load_all_models()
    except Exception as e:
        st.error(f"Error cargando modelos: {e}")
        st.stop()

def render_sidebar():
    """Renderiza la barra lateral con configuraciones"""
    with st.sidebar:
        # Selector de página
        page = st.selectbox(
            "🧭 Navegación",
            ["Predicción", "Reentrenamiento"],
            help="Selecciona la funcionalidad que deseas usar"
        )

        st.divider()

        # Botón para descargar el manual
        manual_path = Path("Manual de Usuario de FetalGrowth AI.pdf")
        if manual_path.exists():
            with open(manual_path, "rb") as file:
                st.download_button(
                    label="📖 Descargar Manual",
                    data=file,
                    file_name="Manual de Usuario de FetalGrowth AI.pdf",
                    mime="application/pdf",
                    help="Descarga el manual de usuario en formato PDF"
                )
        else:
            st.warning("⚠️ Manual no encontrado. Contacte al administrador.")

        st.divider()

        # Configuraciones específicas para la página de predicción
        if page == "Predicción":
            st.header("⚙️ Configuración")

            input_method = st.radio(
                "Método de entrada:",
                ["📝 Manual", "📂 Archivo"]
            )

            chart_type = st.selectbox(
                "Tipo de gráfico:",
                ["Barra", "Pie"]
            )

            threshold = st.slider(
                "Umbral de probabilidad (%)",
                min_value=0,
                max_value=100,
                value=0,
                step=5,
                help="Solo se mostrarán predicciones con probabilidad mayor al umbral"
            )

            return page, input_method, chart_type, threshold

        else:
            # Para la página de reentrenamiento no necesitamos estas configuraciones
            return page, None, None, None

# Resto del código de main.py permanece igual
def get_input_data(input_method):
    """Obtiene los datos de entrada según el método seleccionado"""
    if input_method == "📝 Manual":
        with st.expander("📝 Entrada Manual de Datos"):
            data = get_manual_input()

            # Validar datos completos
            if data.isnull().values.any() or (data == 0).values.any():
                st.warning("⚠️ Todos los campos deben estar completos y mayores a cero.")
                return None
            return data

    else:  # Archivo
        with st.expander("📂 Carga de Archivo"):
            # Mostrar instrucciones
            show_file_instructions()

            uploaded_file = st.file_uploader(
                "Subir archivo (CSV o Excel)",
                type=["csv", "xlsx"]
            )

            if not uploaded_file:
                return None

            # Determinar tipo de archivo
            file_type = 'csv' if uploaded_file.name.endswith('.csv') else 'xlsx'
            data = load_data(uploaded_file, file_type)

            if data is None:
                return None

            # Validar estructura de datos
            valid, msg = validate_input_data(data)
            if not valid:
                st.error(msg)
                return None

            st.success("✅ Archivo cargado correctamente")
            st.dataframe(data, height=150, use_container_width=True)
            return data

def make_predictions(data, models, threshold):
    """Realiza las predicciones y filtra por umbral"""
    try:
        with st.spinner("🔍 Analizando datos..."):
            # Procesar datos
            scaled_data = process_data(data, models['scaler'])

            if scaled_data is None:
                st.error("Error procesando los datos")
                return

            # Hacer predicciones
            predicted_classes = models['main'].predict(scaled_data)
            probabilities = models['main'].predict_proba(scaled_data)

            # Filtrar por umbral
            mask = np.max(probabilities, axis=1) * 100 >= threshold

            if not mask.any():
                st.warning("Ninguna predicción supera el umbral establecido.")
                return

            # Mostrar resultados filtrados
            filtered_classes = predicted_classes[mask]
            filtered_probs = probabilities[mask]

            return filtered_classes, filtered_probs

    except Exception as e:
        st.error(f"Error en la predicción: {e}")
        return None

def render_prediction_page(input_method, chart_type, threshold):
    """Renderiza la página de predicción"""
    # Título
    st.title("FetalGrowth AI - Predicción")
    st.subheader("**Interfaz con modelo de ML integrado para predecir alteraciones en el crecimiento fetal en ratas Wistar**")

    # Cargar modelos
    models = load_models()

    # Obtener datos de entrada
    input_data = get_input_data(input_method)

    # Botón de predicción
    if st.button("🔍 Predecir", type="primary", disabled=input_data is None):
        results = make_predictions(input_data, models, threshold)

        if results:
            filtered_classes, filtered_probs = results

            # Mostrar resultados
            show_results(filtered_classes, filtered_probs, input_data)

            # Visualización
            with st.expander("📊 Visualización de Resultados"):
                render_distribution_chart(filtered_classes, chart_type)

def render_info_banner():
    """Renderiza banner informativo sobre actualizaciones"""
    if 'model_updated' in st.session_state and st.session_state['model_updated']:
        st.success("✅ Modelo actualizado recientemente. Los cambios están activos en las predicciones.")
        if st.button("Ocultar notificación"):
            st.session_state['model_updated'] = False
            st.rerun()

def main():
    """Función principal de la aplicación"""
    # Configuración lateral y obtener página seleccionada
    page_config = render_sidebar()
    page = page_config[0]

    # Mostrar banner informativo si es necesario
    render_info_banner()

    # Renderizar página según selección
    if page == "Predicción":
        input_method, chart_type, threshold = page_config[1], page_config[2], page_config[3]
        render_prediction_page(input_method, chart_type, threshold)

    elif page == "Reentrenamiento":
        render_retrain_page()

if __name__ == "__main__":
    main()