"""
Configuración central para la aplicación FetalGrowth AI
"""

from pathlib import Path

# Configuración de directorios
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
BACKUP_DIR = MODELS_DIR / "backups"

# Rutas de archivos principales
ORIGINAL_DATA_FILE = "Datos originales.xlsx"  # Cambia este nombre por el de tu archivo
ORIGINAL_DATA_PATH = DATA_DIR / ORIGINAL_DATA_FILE

# Archivos de modelo
MODEL_FILES = {
    'main': 'modelo_RandomForest_original.pkl',
    'scaler': 'MS.pkl'
}

# Configuración de datos
FEATURE_COLUMNS = [
    "Grupo", "glicdia14", "glicdia20", "Creat", "Col", 
    "Trig", "VLDL", "Insul", "hemglic"
]

TARGET_COLUMN = "Clasif fetos"
ALL_REQUIRED_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]

# Configuración de modelo
TRAIN_TEST_SPLIT_RATIO = 0.3  # 30% para prueba, 70% para entrenamiento
RANDOM_STATE = 42

# Configuración de hiperparámetros para GridSearchCV
PARAM_GRID = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'max_features': ['sqrt', 'log2'],
    'min_samples_split': [2],
    'min_samples_leaf': [1, 2, 5]
}

# Configuración de validación cruzada
CV_FOLDS = 5

# Mapeo de clases
CLASS_NAMES = {
    1: "PEG (Pequeño para la Edad Gestacional)",
    2: "AEG (Apropiado para la Edad Gestacional)", 
    3: "GEG (Grande para la Edad Gestacional)"
}

CLASS_LABELS = ['PEG', 'AEG', 'GEG']

# Configuración de interfaz
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

# Función para crear directorios necesarios
def ensure_directories():
    """Crea los directorios necesarios si no existen"""
    DATA_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    BACKUP_DIR.mkdir(exist_ok=True)

# Función para verificar configuración
def verify_setup():
    """Verifica que la configuración esté correcta"""
    missing_files = []

    # Verificar que existe el archivo de datos original
    if not ORIGINAL_DATA_PATH.exists():
        missing_files.append(str(ORIGINAL_DATA_PATH))

    # Verificar que existen los archivos de modelo
    for model_name, filename in MODEL_FILES.items():
        model_path = MODELS_DIR / filename
        if not model_path.exists():
            missing_files.append(str(model_path))

    return missing_files