import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Configuración de página
st.set_page_config(
    page_title="Predicción de Alteraciondes del Crecimiento Fetal",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título de la aplicación
st.title("Predicción de Alteraciones del Crecimiento Fetal")
st.markdown(
    "**Herramienta basada en Machine Learning para predecir alteraciones en el crecimiento fetal en ratas Wistar.**"
)

# Cargar componentes entrenados
model = joblib.load('modelo_RandomForest.pkl')
imputer = joblib.load('imputer.pkl')
scaler = joblib.load('scaler.pkl')
rfr_models = {
    'Creat': joblib.load('modelo_creat.pkl'),
    'Col': joblib.load('modelo_col.pkl'),
    'Trig': joblib.load('modelo_trig.pkl'),
    'VLDL': joblib.load('modelo_vldl.pkl')
}

# Función de preprocesamiento
def preprocess_data(input_df):
    try:
        req_columns = [
            'Grupo', 'glicdia14', 'glicdia20', 'Creat', 'Col', 'Trig', 'VLDL',
            'Insul', 'hemglic'
        ]
        df = input_df[req_columns].copy()

        df.dropna(how='all', inplace=True)
        if df.empty:
            st.warning(
                "⚠️ El archivo no contiene datos válidos después de eliminar filas vacías."
            )
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

# 🚀 UX: Guiado en sidebar
with st.sidebar:
    input_method = st.radio("*Seleccionar modo de entrada:*",
                           ["Entrada Manual", "Subir Archivo"])

input_data = None

if input_method == "📝 Manual":
    st.header("Entrada Manual de Datos")
    col1, col2 = st.columns(2)
    
    # 🚀 UX: Agrupación visual de inputs
    with col1:
        st.markdown("<div class='input-section'>", unsafe_allow_html=True)
        grupo = st.number_input("Grupo")
        glicdia14 = st.number_input("Glucosa día 14")
        glicdia20 = st.number_input("Glucosa día 20")
        creat = st.number_input("Creatinina")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='input-section'>", unsafe_allow_html=True)
        col = st.number_input("Colesterol")
        trig = st.number_input("Triglicéridos")
        vldl = st.number_input("VLDL")
        insul = st.number_input("Insulina")
        hemglic = st.number_input("Hemoglobina Glicosilada")
        st.markdown("</div>", unsafe_allow_html=True)

    input_data = pd.DataFrame(
        [[grupo, glicdia14, glicdia20, creat, col, trig, vldl, insul, hemglic]],
        columns=pd.Index([
            'Grupo', 'glicdia14', 'glicdia20', 'Creat', 'Col', 'Trig', 'VLDL',
            'Insul', 'hemglic'
        ]))

else:
    st.header("Carga de Archivo")
    uploaded_file = st.file_uploader("Subir archivo (CSV o Excel)",
                                    type=["csv", "xlsx"])

    if uploaded_file:
        # 🚀 UX: Vista previa de datos
        st.success(f"✅ Archivo cargado correctamente: {uploaded_file.name}")
        with st.expander("🔍 Vista previa de datos"):
            if uploaded_file.name.endswith('.csv'):
                input_data = pd.read_csv(uploaded_file)
            else:
                input_data = pd.read_excel(uploaded_file)
            st.dataframe(input_data, height=150, use_container_width=True)

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
