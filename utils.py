import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Optional

# Configuración de columnas requeridas
REQUIRED_COLUMNS = [
    'Grupo', 'glicdia14', 'glicdia20', 'Creat', 'Col', 'Trig', 'VLDL', 'Insul', 'hemglic'
]

# Mapeo de clases a nombres descriptivos
CLASS_NAMES = {
    1: "PEG (Pequeño para la Edad Gestacional)",
    2: "AEG (Apropiado para la Edad Gestacional)",
    3: "GEG (Grande para la Edad Gestacional)"
}

# Configuración de campos para entrada manual
FIELD_CONFIG = {
    'Grupo':     {'min': 1, 'max': 2, 'step': 1},
    'glicdia14': {'min': 0.0, 'max': 35.0, 'step': 0.1},
    'glicdia20': {'min': 0.0, 'max': 35.0, 'step': 0.1},
    'Creat':     {'min': 0.0, 'max': 618.0, 'step': 0.1},
    'Col':       {'min': 0.0, 'max': 16.0, 'step': 0.1},
    'Trig':      {'min': 0.0, 'max': 16.0, 'step': 0.1},
    'VLDL':      {'min': 0.0, 'max': 7.0, 'step': 0.1},
    'Insul':     {'min': 0.0, 'max': 500.0, 'step': 0.1},
    'hemglic':   {'min': 0.0, 'step': 0.1}
}

FIELD_LABELS = {
    'Grupo': 'Grupo',
    'glicdia14': 'Glucemia día 14',
    'glicdia20': 'Glucemia día 20',
    'Creat': 'Creatinina',
    'Col': 'Colesterol',
    'Trig': 'Triglicéridos',
    'VLDL': 'VLDL',
    'Insul': 'Insulina',
    'hemglic': 'Hemoglobina Glicosilada'
}

def load_data(file, file_type: str) -> Optional[pd.DataFrame]:
    """
    Carga datos desde un archivo.

    Args:
        file: Archivo cargado
        file_type: Tipo de archivo ('csv' o 'xlsx')

    Returns:
        DataFrame con los datos o None si hay error
    """
    try:
        if file_type == 'csv':
            return pd.read_csv(file)
        elif file_type == 'xlsx':
            return pd.read_excel(file)
        else:
            st.error("Tipo de archivo no soportado")
            return None
    except Exception as e:
        st.error(f"Error al cargar el archivo: {e}")
        return None

def validate_input_data(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Valida la estructura y tipos de datos del DataFrame.

    Args:
        df: DataFrame a validar

    Returns:
        Tupla (es_válido, mensaje_error)
    """
    # Verificar columnas faltantes
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        return False, f"Faltan columnas requeridas: {', '.join(missing)}"

    # Verificar tipos numéricos
    non_numeric = []
    for col in REQUIRED_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            non_numeric.append(col)

    if non_numeric:
        return False, f"Columnas no numéricas: {', '.join(non_numeric)}"

    return True, ""

def show_file_instructions():
    """
    Muestra instrucciones claras y atractivas para la carga de archivos.
    """
    st.markdown("### 📋 Instrucciones para el Archivo")

    # Información principal
    st.info("""
    **Su archivo debe contener exactamente estas 9 columnas con estos nombres:**
    """)

    # Crear tabla de columnas requeridas con descripciones
    col_info = pd.DataFrame({
        '🏷️ Nombre de Columna': REQUIRED_COLUMNS,
        '📝 Descripción': [
            'Número de grupo experimental',
            'Glucemia en el día 14',
            'Glucemia en el día 20', 
            'Nivel de creatinina',
            'Nivel de colesterol',
            'Nivel de triglicéridos',
            'Lipoproteínas VLDL',
            'Nivel de insulina',
            'Hemoglobina glicosilada'
        ]
    })

    st.dataframe(col_info, use_container_width=True, hide_index=True)

    # Consejos importantes
    col1, col2 = st.columns(2)

    with col1:
        st.success("""
        **✅ Formato Correcto:**
        • Archivo CSV o Excel (.xlsx)
        • Nombres exactos de columnas
        • Solo valores numéricos
        • Sin filas completamente vacías
        """)

    with col2:
        st.warning("""
        **⚠️ Evite:**
        • Cambiar nombres de columnas
        • Incluir texto en datos numéricos
        • Dejar columnas vacías
        • Usar caracteres especiales
        """)


def show_results(predicted_classes: np.ndarray, probabilities: np.ndarray, df: pd.DataFrame):
    """
    Muestra los resultados de predicción en formato tabular y métricas.

    Args:
        predicted_classes: Clases predichas
        probabilities: Probabilidades de cada clase
        df: DataFrame original (para referencia)
    """
    # Crear DataFrame de resultados
    results_df = pd.DataFrame({
        'Categoría Predicha': [CLASS_NAMES[cls] for cls in predicted_classes],
        'Prob. PEG (%)': probabilities[:, 0] * 100,
        'Prob. AEG (%)': probabilities[:, 1] * 100,
        'Prob. GEG (%)': probabilities[:, 2] * 100
    })

    # Mostrar tabla de resultados
    st.subheader("📊 Resultados Individuales")
    st.dataframe(
        results_df.style.format({
            'Prob. PEG (%)': '{:.2f}%',
            'Prob. AEG (%)': '{:.2f}%',
            'Prob. GEG (%)': '{:.2f}%'
        }),
        use_container_width=True
    )

    # Mostrar métricas agregadas
    st.subheader("📈 Distribución General")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total PEG", np.sum(predicted_classes == 1))
    with col2:
        st.metric("Total AEG", np.sum(predicted_classes == 2))
    with col3:
        st.metric("Total GEG", np.sum(predicted_classes == 3))

def get_manual_input() -> pd.DataFrame:
    """
    Crea interface para entrada manual de datos.

    Returns:
        DataFrame con los datos ingresados
    """
    st.subheader("Entrada Manual de Datos")

    # Dividir campos en dos columnas
    col1, col2 = st.columns(2)
    data = {}

    fields = list(REQUIRED_COLUMNS)
    mid_point = len(fields) // 2

    with col1:
        for field in fields[:mid_point]:
            config = FIELD_CONFIG.get(field, {'min': 0.0, 'step': 0.1})
            label = FIELD_LABELS.get(field, field)

            value = st.number_input(
                label,
                min_value=config['min'],
                max_value=config.get('max'),
                step=config['step']
            )
            data[field] = [value]

    with col2:
        for field in fields[mid_point:]:
            config = FIELD_CONFIG.get(field, {'min': 0.0, 'step': 0.1})
            label = FIELD_LABELS.get(field, field)

            value = st.number_input(
                label,
                min_value=config['min'],
                max_value=config.get('max'),
                step=config['step']
            )
            data[field] = [value]

    return pd.DataFrame(data)


def render_distribution_chart(predicted_classes: np.ndarray, chart_type: str):
    """
    Renderiza gráfico de distribución de clases predichas.

    Args:
        predicted_classes: Array con las clases predichas
        chart_type: Tipo de gráfico ('Barra' o 'Pie')
    """
    class_labels = ['PEG', 'AEG', 'GEG']
    counts = [
        np.sum(predicted_classes == 1),
        np.sum(predicted_classes == 2),
        np.sum(predicted_classes == 3)
    ]

    # Crear DataFrame para visualización
    dist_data = pd.DataFrame({
        'Categoría': class_labels,
        'Casos': counts
    })

    if chart_type == "Barra":
        st.bar_chart(dist_data.set_index('Categoría'))
    else:  # Pie chart
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(
            dist_data['Casos'],
            labels=dist_data['Categoría'],
            autopct='%1.1f%%',
            startangle=90
        )
        ax.set_title('Distribución de Categorías Predichas')
        st.pyplot(fig)