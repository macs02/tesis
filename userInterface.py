import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# ====================== 🎨 ESTILOS MEJORADOS ======================
st.markdown("""
<style>
/* 🔷 Fuente elegante */
@import url('https://fonts.googleapis.com/css2?family=Lato:wght@300;400;700&display=swap');
* {
    font-family: 'Lato', sans-serif !important;
}

/* 🔷 Modo Claro */
[data-testid="stAppViewContainer"] {
    background-color: #f4f7f9;
    color: #1d3557;
}

/* 🔷 Modo Oscuro */
.dark-mode {
    background-color: #1a1a1a !important;
    color: #e0e0e0 !important;
}

/* 🔹 Contenedores principales */
[data-testid="stSidebar"] {
    background-color: #e8f0f2 !important;
}
.dark-mode [data-testid="stSidebar"] {
    background-color: #222831 !important;
}

/* 🔹 Inputs */
.stTextInput, .stNumberInput, .stSelectbox {
    border-radius: 8px !important;
    border: 1px solid #6c757d !important;
}

/* 🔹 Botón de predicción */
.stButton>button {
    background-color: #0077b6 !important;
    color: white !important;
    border-radius: 8px !important;
    transition: all 0.3s ease-in-out !important;
}
.stButton>button:hover {
    background-color: #023e8a !important;
}

/* 🔹 Métricas */
[data-testid="metric-container"] {
    background: #e3f2fd !important;
    border-radius: 12px !important;
    padding: 15px !important;
}
.dark-mode [data-testid="metric-container"] {
    background: #212529 !important;
}

/* 🔹 Tablas */
[data-testid="stTable"] {
    background: white !important;
}
.dark-mode [data-testid="stTable"] {
    background: #2b2b2b !important;
}
</style>
""", unsafe_allow_html=True)

# ====================== 🌗 SWITCH DE TEMA ======================
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

def toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode

# Botón en sidebar para alternar tema
with st.sidebar:
    st.button("🌗 Alternar Tema", on_click=toggle_theme)
    if st.session_state.dark_mode:
        st.markdown("<style>[data-testid='stAppViewContainer'] {background-color: #1a1a1a !important; color: white !important;}</style>", unsafe_allow_html=True)
    else:
        st.markdown("<style>[data-testid='stAppViewContainer'] {background-color: #f4f7f9 !important; color: #1d3557 !important;}</style>", unsafe_allow_html=True)

# ====================== 🩺 TÍTULO Y DESCRIPCIÓN ======================
st.title("🔬 Predicción de Alteraciones del Crecimiento Fetal")
st.markdown(
    "**🧬 Herramienta basada en Machine Learning para predecir alteraciones en el crecimiento fetal en ratas Wistar.**"
)

# ====================== ⚙️ CARGA DE MODELOS ======================
model = joblib.load('modelo_RandomForest.pkl')
imputer = joblib.load('imputer.pkl')
scaler = joblib.load('scaler.pkl')
rfr_models = {
    'Creat': joblib.load('modelo_creat.pkl'),
    'Col': joblib.load('modelo_col.pkl'),
    'Trig': joblib.load('modelo_trig.pkl'),
    'VLDL': joblib.load('modelo_vldl.pkl')
}

# ====================== 🔄 FUNCIÓN DE PREPROCESAMIENTO ======================
def preprocess_data(input_df):
    try:
        req_columns = [
            'Grupo', 'glicdia14', 'glicdia20', 'Creat', 'Col', 'Trig', 'VLDL',
            'Insul', 'hemglic'
        ]
        df = input_df[req_columns].copy()
        df.dropna(how='all', inplace=True)
        if df.empty:
            st.warning("⚠️ No hay datos válidos después de eliminar filas vacías.")
            return None

        df = pd.DataFrame(imputer.transform(df), columns=df.columns)

        for col in ['Creat', 'Col', 'Trig', 'VLDL']:
            X = df.drop(col, axis=1)
            df[col] = rfr_models[col].predict(X)

        scaled_data = scaler.transform(df)
        return pd.DataFrame(scaled_data, columns=df.columns)

    except Exception as e:
        st.error(f"Error en preprocesamiento: {str(e)}")
        return None

# ====================== 🎛️ ENTRADA DE DATOS ======================
with st.sidebar:
    input_method = st.radio("📥 Selecciona el método de entrada:", ["📝 Manual", "📂 Archivo"])

input_data = None

if input_method == "📝 Manual":
    st.header("📋 Entrada Manual de Datos")
    col1, col2 = st.columns(2)

    with col1:
        grupo = st.number_input("Grupo")
        glicdia14 = st.number_input("Glucosa día 14")
        glicdia20 = st.number_input("Glucosa día 20")
        creat = st.number_input("Creatinina")

    with col2:
        col = st.number_input("Colesterol")
        trig = st.number_input("Triglicéridos")
        vldl = st.number_input("VLDL")
        insul = st.number_input("Insulina")
        hemglic = st.number_input("Hemoglobina Glicosilada")

    input_data = pd.DataFrame(
        [[grupo, glicdia14, glicdia20, creat, col, trig, vldl, insul, hemglic]],
        columns=['Grupo', 'glicdia14', 'glicdia20', 'Creat', 'Col', 'Trig', 'VLDL', 'Insul', 'hemglic'])

else:
    st.header("📂 Carga de Archivo")
    uploaded_file = st.file_uploader("📥 Subir archivo CSV o Excel", type=["csv", "xlsx"])

    if uploaded_file:
        st.success(f"✅ Archivo cargado correctamente: {uploaded_file.name}")
        with st.expander("🔍 Vista previa de datos"):
            if uploaded_file.name.endswith('.csv'):
                input_data = pd.read_csv(uploaded_file)
            else:
                input_data = pd.read_excel(uploaded_file)
            st.dataframe(input_data)
            
# 🚀 UX: Botón mejorado con feedback
original_button = st.button("Predecir")
if original_button and input_data is not None:
    try:
        with st.spinner("🔍 Analizando datos. Por favor espere..."):
            input_data_filtered = input_data.dropna(how='all')

            if input_data_filtered.empty:
                st.warning(
                    "⚠️ El archivo no contiene datos válidos. Por favor, revisa los datos."
                )
            else:
                processed_data = preprocess_data(input_data)

                if processed_data is not None:
                    predicted_classes = model.predict(processed_data)
                    probabilities = model.predict_proba(processed_data)

                    class_names = {
                        1: "PEG (Pequeño para la Edad Gestacional)",
                        2: "AEG (Apropiado para la Edad Gestacional)",
                        3: "GEG (Grande para la Edad Gestacional)"
                    }

                    results_df = pd.DataFrame({
                        'Categoría Predicha': [class_names[x] for x in predicted_classes],
                        'Probabilidad PEG (%)': probabilities[:, 0] * 100,
                        'Probabilidad AEG (%)': probabilities[:, 1] * 100,
                        'Probabilidad GEG (%)': probabilities[:, 2] * 100
                    })

                    st.subheader("Resultados Individuales")
                    st.dataframe(results_df.style.format({
                        'Probabilidad PEG (%)': '{:.2f}%',
                        'Probabilidad AEG (%)': '{:.2f}%',
                        'Probabilidad GEG (%)': '{:.2f}%'
                    }), use_container_width=True)

                    st.subheader("Distribución General")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total PEG", f"{np.sum(predicted_classes == 1)}")
                    with col2:
                        st.metric("Total AEG", f"{np.sum(predicted_classes == 2)}")
                    with col3:
                        st.metric("Total GEG", f"{np.sum(predicted_classes == 3)}")

                    st.markdown("---")
                    peg_cases = np.sum(predicted_classes == 1)
                    aeg_cases = np.sum(predicted_classes == 2)
                    geg_cases = np.sum(predicted_classes == 3)

                    # 🚀 UX: Gráfico de distribución
                    with st.expander("📊 Distribución de Resultados", expanded=True):
                        dist_data = pd.DataFrame({
                            'Categoría': ['PEG', 'AEG', 'GEG'],
                            'Casos': [peg_cases, aeg_cases, geg_cases]
                        })
                        st.bar_chart(dist_data.set_index('Categoría'), use_container_width=True)
                        

    except Exception as e:
        st.error(f"Error en la predicción: {str(e)}")
