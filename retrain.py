import pandas as pd
import numpy as np
import streamlit as st
import joblib
import os
import shutil
from datetime import datetime
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from typing import Tuple, Optional
from config import (
    ORIGINAL_DATA_PATH, MODELS_DIR, BACKUP_DIR, MODEL_FILES,
    FEATURE_COLUMNS, TARGET_COLUMN, ALL_REQUIRED_COLUMNS,
    PARAM_GRID, CV_FOLDS, TRAIN_TEST_SPLIT_RATIO, RANDOM_STATE
)

def ensure_directories():
    """Asegura que existan los directorios necesarios"""
    MODELS_DIR.mkdir(exist_ok=True)
    BACKUP_DIR.mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)

def validate_new_data(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Valida que los nuevos datos tengan la estructura correcta.

    Args:
        df: DataFrame con los nuevos datos

    Returns:
        Tupla (es_válido, mensaje)
    """
    # Verificar columnas requeridas
    missing_cols = [col for col in ALL_REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        return False, f"Faltan columnas requeridas: {', '.join(missing_cols)}"

    # Verificar que no haya valores nulos en columnas críticas
    null_counts = df[ALL_REQUIRED_COLUMNS].isnull().sum()
    if null_counts.any():
        cols_with_nulls = null_counts[null_counts > 0].index.tolist()
        return False, f"Hay valores nulos en: {', '.join(cols_with_nulls)}"

    # Verificar tipos de datos numéricos
    non_numeric_cols = []
    for col in ALL_REQUIRED_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            non_numeric_cols.append(col)

    if non_numeric_cols:
        return False, f"Columnas no numéricas: {', '.join(non_numeric_cols)}"

    # Verificar valores válidos en la columna objetivo
    valid_targets = [1, 2, 3]  # PEG, AEG, GEG
    invalid_targets = df[~df[TARGET_COLUMN].isin(valid_targets)]
    if not invalid_targets.empty:
        return False, f"La columna '{TARGET_COLUMN}' debe contener solo valores 1, 2, o 3"

    return True, "Datos válidos"

def load_original_data() -> Optional[pd.DataFrame]:
    """
    Carga la base de datos original.

    Returns:
        DataFrame con los datos originales o None si hay error
    """
    try:
        if not os.path.exists(ORIGINAL_DATA_PATH):
            return None
        return pd.read_excel(ORIGINAL_DATA_PATH)
    except Exception as e:
        st.error(f"Error cargando datos originales: {e}")
        return None

def combine_and_save_data(original_df: pd.DataFrame, new_df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Combina los datos originales con los nuevos datos, elimina duplicados,
    y guarda el resultado en el archivo original de forma permanente.

    Args:
        original_df: DataFrame con datos originales
        new_df: DataFrame con nuevos datos

    Returns:
        Tupla (DataFrame combinado, número de filas nuevas agregadas)
    """
    # Seleccionar solo las columnas necesarias
    original_clean = original_df[ALL_REQUIRED_COLUMNS].copy()

    # Para los nuevos datos, omitir la primera fila si parece ser encabezados
    new_clean = new_df[ALL_REQUIRED_COLUMNS].copy()

    # Detectar si la primera fila contiene nombres de columnas en lugar de datos
    first_row = new_clean.iloc[0]
    if any(isinstance(val, str) and val in ALL_REQUIRED_COLUMNS for val in first_row.values):
        st.info("Se detectó fila de encabezados en nuevos datos, eliminándola...")
        new_clean = new_clean.iloc[1:].reset_index(drop=True)

    # Verificar que aún tengamos datos después de eliminar encabezados
    if new_clean.empty:
        raise ValueError("No hay datos válidos después de eliminar encabezados duplicados")

    # Convertir todas las columnas a numéricas (por si había strings)
    for col in ALL_REQUIRED_COLUMNS:
        new_clean[col] = pd.to_numeric(new_clean[col], errors='coerce')

    # Eliminar filas con valores NaN que pudieron haberse creado
    new_clean = new_clean.dropna()

    if new_clean.empty:
        raise ValueError("No hay datos válidos después de la conversión numérica")

    # Combinar datos
    combined_df = pd.concat([original_clean, new_clean], ignore_index=True)


    # Calcular cuántas filas nuevas se agregaron realmente
    new_rows_added = len(combined_df) - len(original_clean)

    # Guardar el DataFrame combinado en el archivo original (PERMANENTE)
    try:
        # Crear backup del archivo original antes de modificarlo
        backup_original_file()

        # Guardar los datos combinados
        combined_df.to_excel(ORIGINAL_DATA_PATH, index=False)
        st.success(f"✅ Archivo original actualizado. Se agregaron {new_rows_added} nuevas filas.")

    except Exception as e:
        st.error(f"Error guardando datos combinados: {e}")
        raise

    return combined_df, new_rows_added

def backup_original_file():
    """Crea un backup del archivo de datos original"""
    if ORIGINAL_DATA_PATH.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"data_backup_{timestamp}.xlsx"
        backup_path = BACKUP_DIR / backup_name
        shutil.copy2(ORIGINAL_DATA_PATH, backup_path)
        st.info(f"Backup de datos originales creado: {backup_name}")

def backup_current_model() -> str:
    """
    Crea un backup del modelo actual.

    Returns:
        Nombre del backup creado
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"model_backup_{timestamp}"
    backup_path = BACKUP_DIR / backup_name
    backup_path.mkdir(exist_ok=True)

    # Copiar archivos del modelo actual
    model_files = list(MODEL_FILES.values())

    for file_name in model_files:
        src_path = MODELS_DIR / file_name
        if src_path.exists():
            dst_path = backup_path / file_name
            shutil.copy2(src_path, dst_path)

    return backup_name

def train_new_model(combined_data: pd.DataFrame) -> Tuple[bool, str, dict]:
    """
    Entrena un nuevo modelo con los datos combinados.

    Args:
        combined_data: DataFrame con todos los datos

    Returns:
        Tupla (éxito, mensaje, métricas)
    """
    try:
        # Separar características y objetivo
        X = combined_data[FEATURE_COLUMNS]
        y = combined_data[TARGET_COLUMN]

        # Dividir en entrenamiento y prueba (70/30)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TRAIN_TEST_SPLIT_RATIO, random_state=RANDOM_STATE, stratify=y
        )

        # Normalizar características
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Configurar modelo Random Forest
        rf = RandomForestClassifier(random_state=RANDOM_STATE)

        # Validación cruzada con búsqueda de hiperparámetros
        grid_search = GridSearchCV(
            estimator=rf, 
            param_grid=PARAM_GRID, 
            cv=CV_FOLDS, 
            scoring='accuracy', 
            n_jobs=-1
        )

        grid_search.fit(X_train_scaled, y_train)

        # Obtener el mejor modelo
        best_model = grid_search.best_estimator_

        # Hacer predicciones en el conjunto de prueba
        y_pred = best_model.predict(X_test_scaled)

        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Calcular matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if cm.shape[0] > 1 else None

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'best_params': grid_search.best_params_,
            'confusion_matrix': cm,
            'train_size': len(X_train),
            'test_size': len(X_test)
        }

        # Guardar el nuevo modelo
        model_files = list(MODEL_FILES.values())
        joblib.dump(best_model, MODELS_DIR / model_files[0])  # modelo_Random_Forest.pkl
        joblib.dump(scaler, MODELS_DIR / model_files[1])      # MS.pkl

        return True, "Modelo entrenado exitosamente", metrics

    except Exception as e:
        return False, f"Error durante el entrenamiento: {str(e)}", {}

def restore_model(backup_name: str) -> Tuple[bool, str]:
    """
    Restaura un modelo desde un backup.

    Args:
        backup_name: Nombre del backup a restaurar

    Returns:
        Tupla (éxito, mensaje)
    """
    try:
        backup_path = BACKUP_DIR / backup_name

        if not backup_path.exists():
            return False, f"Backup {backup_name} no encontrado"

        model_files = list(MODEL_FILES.values())

        for file_name in model_files:
            src_path = backup_path / file_name
            dst_path = MODELS_DIR / file_name

            if src_path.exists():
                shutil.copy2(src_path, dst_path)
            else:
                return False, f"Archivo {file_name} no encontrado en el backup"

        return True, f"Modelo restaurado desde {backup_name}"

    except Exception as e:
        return False, f"Error restaurando modelo: {str(e)}"

def delete_model_backup(backup_name: str) -> Tuple[bool, str]:
    """
    Elimina un backup de modelo específico.

    Args:
        backup_name: Nombre del backup a eliminar

    Returns:
        Tupla (éxito, mensaje)
    """
    try:
        backup_path = BACKUP_DIR / backup_name

        if not backup_path.exists():
            return False, f"Backup {backup_name} no encontrado"

        # Eliminar directorio completo
        shutil.rmtree(backup_path)
        return True, f"Backup {backup_name} eliminado exitosamente"

    except Exception as e:
        return False, f"Error eliminando backup: {str(e)}"

def get_available_backups() -> list:
    """
    Obtiene la lista de backups disponibles.

    Returns:
        Lista de nombres de backups ordenados por fecha (más recientes primero)
    """
    if not BACKUP_DIR.exists():
        return []

    backups = [d.name for d in BACKUP_DIR.iterdir() if d.is_dir() and d.name.startswith("model_backup_")]
    return sorted(backups, reverse=True)  # Más recientes primero

def format_backup_name(backup_name: str) -> str:
    """
    Formatea el nombre del backup para mostrar de forma amigable.

    Args:
        backup_name: Nombre del backup (ej: model_backup_20241127_143052)

    Returns:
        Nombre formateado (ej: 27/11/2024 - 14:30:52)
    """
    try:
        # Extraer fecha y hora del nombre
        parts = backup_name.replace('model_backup_', '').split('_')
        if len(parts) == 2:
            date_part, time_part = parts
            # Convertir formato YYYYMMDD a DD/MM/YYYY
            formatted_date = f"{date_part[6:8]}/{date_part[4:6]}/{date_part[:4]}"
            # Convertir formato HHMMSS a HH:MM:SS
            formatted_time = f"{time_part[:2]}:{time_part[2:4]}:{time_part[4:6]}"
            return f"{formatted_date} - {formatted_time}"
    except:
        pass

    # Si no se puede formatear, devolver el nombre original
    return backup_name.replace('model_backup_', '')

def display_metrics(metrics: dict):
    """
    Muestra las métricas del modelo en Streamlit.

    Args:
        metrics: Diccionario con las métricas
    """
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Precisión", f"{metrics['accuracy']:.4f}")
    with col2:
        st.metric("Precision", f"{metrics['precision']:.4f}")
    with col3:
        st.metric("Recall", f"{metrics['recall']:.4f}")
    with col4:
        st.metric("F1-Score", f"{metrics['f1_score']:.4f}")

    if metrics.get('specificity'):
        st.metric("Especificidad", f"{metrics['specificity']:.4f}")

    # Mostrar información adicional
    st.subheader("📊 Información del Entrenamiento")
    col1, col2 = st.columns(2)

    with col1:
        st.info(f"**Datos de entrenamiento:** {metrics['train_size']}")
        st.info(f"**Datos de prueba:** {metrics['test_size']}")

    with col2:
        st.info("**Mejores hiperparámetros:**")
        for param, value in metrics['best_params'].items():
            st.write(f"• {param}: {value}")