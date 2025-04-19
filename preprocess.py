# preprocess.py
import pandas as pd


def validate_and_filter_data(input_df, required_columns):
    """
    Valida que el DataFrame contenga las columnas requeridas, 
    las reordena y elimina filas vacías.
    """
    df = input_df.copy()
    df = df[required_columns]  # fuerza el orden correcto
    df.dropna(how='all', inplace=True)
    return df


def preprocess_data(input_df, imputer, scaler, rfr_models):
    """
    Preprocesa los datos:
    - Valida y filtra las columnas necesarias.
    - Imputa datos faltantes.
    - Corrige variables específicas usando modelos entrenados.
    - Escala los datos.
    """
    try:
        req_columns = [
            'Grupo', 'glicdia14', 'glicdia20', 'Creat', 'Col', 'Trig', 'VLDL',
            'Insul', 'hemglic'
        ]
        df = validate_and_filter_data(input_df, req_columns)

        if df.empty:
            raise ValueError(
                "El archivo no contiene datos válidos después de eliminar filas vacías."
            )

        # Imputación de datos
        df[:] = imputer.transform(df)

        # Predicción de variables específicas
        for col in ['Creat', 'Col', 'Trig', 'VLDL']:
            X = df.drop(col, axis=1)
            df[col] = rfr_models[col].predict(X)

        # Escalado
        scaled_data = scaler.transform(df)
        return pd.DataFrame(scaled_data, columns=df.columns)

    except Exception as e:
        raise RuntimeError(f"Error en preprocesamiento: {str(e)}")
