import pandas as pd
import numpy as np
from typing import Optional

# Columnas requeridas en orden
REQUIRED_COLUMNS = [
    'Grupo', 'glicdia14', 'glicdia20', 'Creat', 'Col', 'Trig', 'VLDL', 'Insul', 'hemglic'
]

def validate_columns(df: pd.DataFrame, required_columns: list) -> pd.DataFrame:
    """
    Valida que el DataFrame tenga las columnas requeridas y las ordena correctamente.

    Args:
        df: DataFrame a validar
        required_columns: Lista de columnas requeridas

    Returns:
        DataFrame con columnas ordenadas

    Raises:
        ValueError: Si faltan columnas requeridas
    """
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columnas faltantes: {missing_cols}")

    # Retornar DataFrame con columnas en el orden correcto
    return df[REQUIRED_COLUMNS].copy()

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia el DataFrame eliminando filas completamente vacías.

    Args:
        df: DataFrame a limpiar

    Returns:
        DataFrame limpio

    Raises:
        ValueError: Si no quedan datos después de la limpieza
    """
    cleaned_df = df.dropna(how='all')

    if cleaned_df.empty:
        raise ValueError("No hay datos válidos después de la limpieza")

    return cleaned_df

def apply_scaling(df: pd.DataFrame, scaler) -> Optional[pd.DataFrame]:
    """
    Aplica escalado a los datos.

    Args:
        df: DataFrame con datos a escalar
        scaler: Objeto scaler entrenado

    Returns:
        DataFrame escalado o None si hay error
    """
    try:
        scaled_values = scaler.transform(df)
        scaled_df = pd.DataFrame(
            scaled_values,
            columns=df.columns,
            index=df.index
        )
        return scaled_df

    except Exception as e:
        print(f"Error en escalado: {e}")
        return None

def process_data(input_df: pd.DataFrame, scaler) -> Optional[pd.DataFrame]:
    """
    Procesa los datos de entrada: validación, limpieza y escalado.

    Args:
        input_df: DataFrame con datos de entrada
        scaler: Objeto scaler entrenado

    Returns:
        DataFrame procesado y escalado, o None si hay error
    """
    try:
        # Validar y ordenar columnas
        validated_df = validate_columns(input_df, REQUIRED_COLUMNS)

        # Limpiar datos
        cleaned_df = clean_data(validated_df)

        # Aplicar escalado
        scaled_df = apply_scaling(cleaned_df, scaler)

        return scaled_df

    except Exception as e:
        print(f"Error procesando datos: {e}")
        return None