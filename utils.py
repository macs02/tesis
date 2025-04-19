# utils.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def show_results(predicted_classes, probabilities):
    """
    Muestra los resultados de la predicción en forma de tabla y métricas.
    """
    # Mapear las clases a sus nombres descriptivos
    class_names = {
        1: "PEG (Pequeño para la Edad Gestacional)",
        2: "AEG (Apropiado para la Edad Gestacional)",
        3: "GEG (Grande para la Edad Gestacional)"
    }

    results_df = pd.DataFrame({
        'Categoría Predicha': [class_names[x] for x in predicted_classes],
        'Probabilidad PEG (%)':
        probabilities[:, 0] * 100,
        'Probabilidad AEG (%)':
        probabilities[:, 1] * 100,
        'Probabilidad GEG (%)':
        probabilities[:, 2] * 100
    })

    st.subheader("Resultados Individuales")
    st.dataframe(results_df.style.format({
        'Probabilidad PEG (%)': '{:.2f}%',
        'Probabilidad AEG (%)': '{:.2f}%',
        'Probabilidad GEG (%)': '{:.2f}%'
    }),
                 use_container_width=True)

    # Mostrar métricas agregadas
    st.subheader("Distribución General")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total PEG", f"{np.sum(predicted_classes == 1)}")
    with col2:
        st.metric("Total AEG", f"{np.sum(predicted_classes == 2)}")
    with col3:
        st.metric("Total GEG", f"{np.sum(predicted_classes == 3)}")



def load_data(file, file_type):
    """
    Carga datos de un archivo en función de su extensión.

    Parámetros:
    - file: objeto file-like cargado mediante st.file_uploader.
    - file_type: str, 'csv' o 'xlsx'.

    Retorna:
    - DataFrame con los datos cargados.

    Lanza:
    - ValueError si el tipo de archivo no es soportado.
    """
    try:
        if file_type == 'csv':
            return pd.read_csv(file)
        elif file_type == 'xlsx':
            return pd.read_excel(file)
        else:
            raise ValueError("Tipo de archivo no soportado")
    except Exception as e:
        st.error(f"Error al cargar el archivo: {e}")
        return None


def validate_columns(df, required_columns):
    """
    Valida que el DataFrame contenga todas las columnas requeridas.

    Parámetros:
    - df: DataFrame a validar.
    - required_columns: lista de nombres de columnas necesarias.

    Retorna:
    - bool: True si todas las columnas están presentes, de lo contrario False.
    """
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        st.error(
            f"Faltan las siguientes columnas requeridas: {', '.join(missing)}")
        return False
    return True



