import streamlit as st
import pandas as pd
from retrain import (
    ensure_directories, validate_new_data, load_original_data,
    combine_and_save_data, backup_current_model, train_new_model, 
    restore_model, get_available_backups, display_metrics,
    delete_model_backup, format_backup_name
)
from config import ALL_REQUIRED_COLUMNS, FEATURE_COLUMNS, TARGET_COLUMN, CLASS_NAMES

def show_data_requirements():
    """Muestra los requisitos de datos para el reentrenamiento"""
    with st.expander("📋 Requisitos de Datos para Reentrenamiento"):
        st.markdown("### Estructura Requerida del Archivo")

        st.info("""
        Su archivo Excel debe contener **exactamente** estas 10 columnas:
        """)

        # Crear tabla con información de columnas
        col_info = pd.DataFrame({
            '🏷️ Nombre de Columna': ALL_REQUIRED_COLUMNS,
            '📝 Descripción': [
                'Número de grupo experimental',
                'Glucemia en el día 14',
                'Glucemia en el día 20',
                'Nivel de creatinina',
                'Nivel de colesterol',
                'Nivel de triglicéridos',
                'Lipoproteínas VLDL',
                'Nivel de insulina',
                'Hemoglobina glicosilada',
                'Clasificación fetal (1=PEG, 2=AEG, 3=GEG)'
            ],
            '🎯 Tipo': ['Numérico'] * 9 + ['Categórico (1, 2, 3)']
        })

        st.dataframe(col_info, use_container_width=True, hide_index=True)

        # Advertencias importantes
        st.warning("""
        **⚠️ IMPORTANTE:**
        • La columna 'Clasif fetos' debe contener únicamente valores 1, 2, o 3
        • Todas las demás columnas deben contener valores numéricos
        • No debe haber celdas vacías en ninguna columna
        • Los nombres de las columnas deben coincidir exactamente
        • La primera fila debe contener los nombres de las columnas (se eliminará automáticamente)
        """)

def render_upload_section():
    """Renderiza la sección de carga de archivos"""
    st.subheader("📂 Cargar Nuevos Datos")

    uploaded_file = st.file_uploader(
        "Seleccione el archivo Excel con los nuevos datos",
        type=['xlsx'],
        help="Solo se aceptan archivos Excel (.xlsx)"
    )

    if uploaded_file is not None:
        try:
            # Cargar datos
            new_data = pd.read_excel(uploaded_file)

            # Mostrar vista previa
            st.success("✅ Archivo cargado correctamente")
            st.subheader("👀 Vista Previa de los Datos")
            st.dataframe(new_data.head(10), use_container_width=True)

            # Mostrar estadísticas básicas
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total de Filas", len(new_data))
            with col2:
                st.metric("Total de Columnas", len(new_data.columns))
            with col3:
                if TARGET_COLUMN in new_data.columns:
                    unique_classes = new_data[TARGET_COLUMN].nunique()
                    st.metric("Clases Únicas", unique_classes)

            # Validar datos
            is_valid, message = validate_new_data(new_data)

            if is_valid:
                st.success(f"✅ {message}")

                # Mostrar distribución de clases
                if TARGET_COLUMN in new_data.columns:
                    st.subheader("📊 Distribución de Clases en Nuevos Datos")
                    class_counts = new_data[TARGET_COLUMN].value_counts().sort_index()
                    class_display_names = {1: 'PEG', 2: 'AEG', 3: 'GEG'}

                    dist_data = pd.DataFrame({
                        'Clase': [class_display_names.get(i, f'Clase {i}') for i in class_counts.index],
                        'Cantidad': class_counts.values
                    })

                    st.bar_chart(dist_data.set_index('Clase'))

                return new_data
            else:
                st.error(f"❌ {message}")
                return None

        except Exception as e:
            st.error(f"Error al procesar el archivo: {str(e)}")
            return None

    return None

def render_training_section(new_data):
    """Renderiza la sección de entrenamiento"""
    st.subheader("🚀 Entrenamiento del Modelo")

    # Cargar datos originales
    original_data = load_original_data()

    if original_data is None:
        st.error("❌ No se pudo cargar la base de datos original. Verifique que el archivo existe.")
        return

    # Mostrar información de combinación
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"**Datos Originales**\n{len(original_data)} registros")
    with col2:
        st.info(f"**Nuevos Datos**\n{len(new_data)} registros")
    with col3:
        estimated_new = len(new_data) - 1  # Menos la fila de encabezados
        st.info(f"**Estimado a Agregar**\n~{estimated_new} registros")

    # Advertencia importante sobre permanencia
    st.warning("""
    ⚠️ **IMPORTANTE**: Los nuevos datos se agregarán **PERMANENTEMENTE** al archivo original. 
    Se creará un backup automático antes de proceder.
    """)

    # Botón de entrenamiento
    if st.button("🔄 Entrenar Nuevo Modelo", type="primary"):
        with st.spinner("🔄 Procesando datos y entrenando modelo... Esto puede tomar varios minutos."):

            try:
                # Crear backup del modelo actual
                backup_name = backup_current_model()
                st.info(f"✅ Backup del modelo creado: {backup_name}")

                # Combinar y guardar datos (PERMANENTE)
                combined_data, new_rows_added = combine_and_save_data(original_data, new_data)
                st.success(f"✅ Se agregaron {new_rows_added} nuevos registros al archivo original")
                st.info(f"📊 Total de registros ahora: {len(combined_data)}")

                # Entrenar nuevo modelo
                success, message, metrics = train_new_model(combined_data)

                if success:
                    st.success(f"🎉 {message}")
                    st.balloons()

                    # Mostrar métricas
                    st.subheader("📈 Métricas del Nuevo Modelo")
                    display_metrics(metrics)

                    # Guardar información del entrenamiento en sesión
                    st.session_state['last_training'] = {
                        'backup_name': backup_name,
                        'metrics': metrics,
                        'timestamp': backup_name.split('_')[-2] + '_' + backup_name.split('_')[-1],
                        'new_rows_added': new_rows_added
                    }

                    # Marcar que el modelo fue actualizado
                    st.session_state['model_updated'] = True

                    st.success("✅ El nuevo modelo está listo para usar en la aplicación principal.")

                    # Limpiar cache de modelos para forzar recarga
                    st.cache_resource.clear()

                else:
                    st.error(f"❌ {message}")

            except Exception as e:
                st.error(f"❌ Error durante el proceso: {str(e)}")

def render_model_management():
    """Renderiza la sección de gestión de modelos"""
    st.subheader("🔧 Gestión de Modelos")

    # Obtener backups disponibles
    backups = get_available_backups()

    if not backups:
        st.info("No hay backups de modelos disponibles.")
        return

    # Mostrar último entrenamiento si está disponible
    if 'last_training' in st.session_state:
        last_training = st.session_state['last_training']
        with st.expander("📊 Información del Último Entrenamiento", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Fecha/Hora", last_training['timestamp'].replace('_', ' - '))
            with col2:
                st.metric("Nuevos Registros", last_training['new_rows_added'])
            with col3:
                if 'metrics' in last_training:
                    accuracy = last_training['metrics'].get('accuracy', 0)
                    st.metric("Precisión", f"{accuracy:.4f}")

    st.write(f"**Modelos disponibles para restaurar:** {len(backups)}")

    # Selector de backup para restaurar
    selected_backup = st.selectbox(
        "Seleccionar modelo para restaurar:",
        backups,
        format_func=format_backup_name,
        help="Seleccione un modelo anterior para restaurar"
    )

    # Botones de acción organizados en columnas
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔄 Restaurar Modelo", type="secondary", use_container_width=True):
            if st.session_state.get('confirm_restore', False):
                with st.spinner("Restaurando modelo..."):
                    success, message = restore_model(selected_backup)

                    if success:
                        st.success(f"✅ {message}")
                        # Limpiar cache para forzar recarga del modelo
                        st.cache_resource.clear()
                        # Limpiar confirmación
                        st.session_state['confirm_restore'] = False
                        # Marcar que el modelo fue actualizado
                        st.session_state['model_updated'] = True
                        st.rerun()
                    else:
                        st.error(f"❌ {message}")
                        st.session_state['confirm_restore'] = False
            else:
                st.session_state['confirm_restore'] = True
                st.warning("⚠️ ¿Confirma restaurar este modelo? Haga clic nuevamente para confirmar.")

    with col2:
        if st.button("🗑️ Eliminar Backup", type="secondary", use_container_width=True):
            if st.session_state.get('confirm_delete', False):
                with st.spinner("Eliminando backup..."):
                    success, message = delete_model_backup(selected_backup)

                    if success:
                        st.success(f"✅ {message}")
                        st.session_state['confirm_delete'] = False
                        st.rerun()
                    else:
                        st.error(f"❌ {message}")
                        st.session_state['confirm_delete'] = False
            else:
                st.session_state['confirm_delete'] = True
                st.warning("⚠️ ¿Confirma eliminar este backup? Haga clic nuevamente para confirmar.")

    with col3:
        if st.button("🔄 Limpiar Cache", type="secondary", use_container_width=True):
            # Limpiar cache de Streamlit para forzar recarga de modelos
            st.cache_resource.clear()
            st.success("✅ Cache limpiado. Los modelos se recargarán automáticamente.")

    # Limpiar confirmaciones si se cambia la selección
    if 'last_selected_backup' not in st.session_state:
        st.session_state['last_selected_backup'] = selected_backup
    elif st.session_state['last_selected_backup'] != selected_backup:
        st.session_state['confirm_restore'] = False
        st.session_state['confirm_delete'] = False
        st.session_state['last_selected_backup'] = selected_backup

def render_data_info():
    """Renderiza información sobre el estado actual de los datos"""
    st.subheader("📊 Estado Actual de los Datos")

    original_data = load_original_data()

    if original_data is not None:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total de Registros", len(original_data))

        with col2:
            if TARGET_COLUMN in original_data.columns:
                unique_classes = original_data[TARGET_COLUMN].nunique()
                st.metric("Clases Únicas", unique_classes)

        with col3:
            st.metric("Columnas", len(original_data.columns))

    else:
        st.error("❌ No se pudo cargar la información de los datos originales")

def render_retrain_page():
    """Función principal para renderizar la página de reentrenamiento"""
    # Configurar página
    st.title("🔄 Reentrenamiento de Modelo")
    st.markdown("---")

    # Asegurar que existan los directorios necesarios
    ensure_directories()

    # Información introductoria
    st.markdown("""
    ### 🎯 Funcionalidad de Reentrenamiento

    Esta página permite **reentrenar** el modelo de clasificación del peso fetal con nuevos datos.
    El proceso incluye:

    1. **Carga de nuevos datos** en formato Excel
    2. **Validación automática** de la estructura de datos  
    3. **Combinación PERMANENTE** con la base de datos original
    4. **Entrenamiento** del modelo con optimización de hiperparámetros
    5. **Backup automático** del modelo anterior
    6. **Gestión intuitiva** para restaurar o eliminar modelos previos
    """)

    # Mostrar estado actual de los datos
    render_data_info()

    st.markdown("---")

    # Mostrar requisitos de datos
    show_data_requirements()

    st.markdown("---")

    # Sección de carga de archivos
    new_data = render_upload_section()

    if new_data is not None:
        st.markdown("---")
        # Sección de entrenamiento
        render_training_section(new_data)

    st.markdown("---")

    # Sección de gestión de modelos
    render_model_management()

    # Información adicional
    with st.expander("ℹ️ Información Adicional"):
        st.markdown("""
        ### 📊 Proceso de Entrenamiento

        - **División de datos**: 70% entrenamiento, 30% prueba
        - **Validación cruzada**: 5-fold cross-validation
        - **Optimización**: Grid search para hiperparámetros
        - **Métricas**: Precisión, Recall, F1-Score, Especificidad

        ### 🔒 Seguridad y Gestión

        - Se crea un backup automático antes de cada entrenamiento
        - **Los datos se guardan PERMANENTEMENTE** en el archivo original
        - Posibilidad de restaurar cualquier modelo anterior de forma intuitiva
        - Opción de eliminar backups innecesarios para ahorrar espacio
        - Validación exhaustiva de datos antes del entrenamiento

        ### ⚠️ Consideraciones Importantes

        - **PERMANENCIA**: Los nuevos datos se agregan al archivo original y no se pueden deshacer
        - El proceso puede tomar varios minutos dependiendo del tamaño de los datos
        - Se eliminan automáticamente filas de encabezados duplicadas
        - Se eliminan registros completamente duplicados (mantiene uno)
        - Asegúrese de que los nuevos datos sean de alta calidad
        - Mantenga copias de seguridad de sus datos importantes

        ### 🔄 Gestión de Modelos

        - **Restaurar**: Vuelve a un modelo anterior (requiere confirmación)
        - **Eliminar**: Borra backups innecesarios (requiere confirmación)  
        - **Cache**: Limpia la memoria para forzar recarga de modelos
        - Los cambios son efectivos inmediatamente en la aplicación principal
        """)

if __name__ == "__main__":
    render_retrain_page()