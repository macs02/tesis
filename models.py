import joblib
import os

def load_model(model_path):
  if os.path.exists(model_path):
    return joblib.load(model_path)
  else:
    raise FileNotFoundError(f'No se encontró el modelo en {model_path}')


def load_all_models(models_dir='models'):
  models = {}
  models['main'] = load_model(f'{models_dir}/modelo_RandomForest.pkl')
  models['imputer'] = load_model(f'{models_dir}/imputer.pkl')
  models['scaler'] = load_model(f'{models_dir}/scaler.pkl')

  # Modelos para imputación o corrección de variables específicas
  models['rfr'] = {
      'Creat': load_model(f'{models_dir}/modelo_creat.pkl'),
      'Col': load_model(f'{models_dir}/modelo_col.pkl'),
      'Trig': load_model(f'{models_dir}/modelo_trig.pkl'),
      'VLDL': load_model(f'{models_dir}/modelo_vldl.pkl')
  }
  return models