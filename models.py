import joblib
import os
from pathlib import Path

# Configuración de rutas de modelos
MODELS_DIR = Path('models')
MODEL_FILES = {
    'main': 'modelo_RandomForest_original.pkl',
    'scaler': 'MS.pkl'
}

def load_model(model_path: Path) -> object:
    """
    Carga un modelo desde el archivo especificado.

    Args:
        model_path: Ruta al archivo del modelo

    Returns:
        Modelo cargado

    Raises:
        FileNotFoundError: Si el archivo no existe
    """
    if not model_path.exists():
        raise FileNotFoundError(f'Modelo no encontrado: {model_path}')

    return joblib.load(model_path)

def load_all_models(models_dir: str = 'models') -> dict:
    """
    Carga todos los modelos necesarios para la aplicación.

    Args:
        models_dir: Directorio donde se encuentran los modelos

    Returns:
        Diccionario con todos los modelos cargados

    Raises:
        FileNotFoundError: Si algún modelo no se encuentra
    """
    models_path = Path(models_dir)
    models = {}

    for model_name, filename in MODEL_FILES.items():
        model_path = models_path / filename
        models[model_name] = load_model(model_path)

    return models