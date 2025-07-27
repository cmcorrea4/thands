import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
import os
import pandas as pd
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Configuración de la página
st.set_page_config(page_title="Clasificador de Posiciones de Manos", page_icon="👋")
st.title("Clasificador de Posiciones de Manos")

# Ruta específica del proyecto
PROJECT_PATH = r"C:\Users\cmcor\Manos"
DATA_PATH = os.path.join(PROJECT_PATH, "data")
MODELS_PATH = os.path.join(PROJECT_PATH, "models")

# Crear directorios para guardar datos y modelos
if not os.path.exists(PROJECT_PATH):
    os.makedirs(PROJECT_PATH)
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)
if not os.path.exists(MODELS_PATH):
    os.makedirs(MODELS_PATH)

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Inicializar variables de estado
if 'features_data' not in st.session_state:
    st.session_state.features_data = []
if 'label' not in st.session_state:
    st.session_state.label = ""
if 'active' not in st.session_state:
    st.session_state.active = False
if 'counter' not in st.session_state:
    st.session_state.counter = 0
if 'start_button' not in st.session_state:
    st.session_state.start_button = False

# Función para extraer características de los puntos clave de la mano
def extract_hand_features(hand_landmarks, handedness):
    """Extrae características relevantes de los puntos clave de la mano"""
    try:
        # Convertir landmarks a un arreglo NumPy
        hand_points = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
        
        # Usar la muñeca como punto de referencia para normalizar
        reference_point = hand_points[mp_hands.HandLandmark.WRIST][:2]  # Solo x, y
        
        # Extraer características normalizadas
        features = []
        
        # 1. Coordenadas normalizadas relativas a la muñeca
        for i, landmark in enumerate(hand_points):
            # Normalizar coordenadas x, y
            norm_x = landmark[0] - reference_point[0]
            norm_y = landmark[1] - reference_point[1]
            # Incluir z para profundidad relativa
            norm_z = landmark[2]
            features.extend([norm_x, norm_y, norm_z])
        
        # 2. Calcular distancias entre puntos específicos
        def calculate_distance_3d(p1, p2):
            """Calcula la distancia euclidiana 3D entre dos puntos"""
            return np.linalg.norm(np.array(p1) - np.array(p2))
        
        # Distancias entre dedos
        thumb_tip = hand_points[mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_points[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_points[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = hand_points[mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = hand_points[mp_hands.HandLandmark.PINKY_TIP]
        
        # Distancias entre puntas de dedos adyacentes
        thumb_index_dist = calculate_distance_3d(thumb_tip, index_tip)
        index_middle_dist = calculate_distance_3d(index_tip, middle_tip)
        middle_ring_dist = calculate_distance_3d(middle_tip, ring_tip)
        ring_pinky_dist = calculate_distance_3d(ring_tip, pinky_tip)
        
        # Distancias a la muñeca
        thumb_wrist_dist = calculate_distance_3d(thumb_tip, hand_points[mp_hands.HandLandmark.WRIST])
        
        features.extend([thumb_index_dist, index_middle_dist, middle_ring_dist, 
                        ring_pinky_dist, thumb_wrist_dist])
        
        # 3. Calcular ángulos entre articulaciones de dedos
        def calculate_angle(a, b, c):
            """Calcula el ángulo entre tres puntos"""
            a = np.array(a)
            b = np.array(b)
            c = np.array(c)
            
            # Vectores
            ba = a - b
            bc = c - b
            
            # Calcular coseno del ángulo
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
            
            return np.degrees(angle)
        
        # Ángulos de cada dedo (MCP-PIP-DIP-TIP)
        for finger_id in [mp_hands.HandLandmark.THUMB_MCP, mp_hands.HandLandmark.INDEX_FINGER_MCP,
                         mp_hands.HandLandmark.MIDDLE_FINGER_MCP, mp_hands.HandLandmark.RING_FINGER_MCP,
                         mp_hands.HandLandmark.PINKY_MCP]:
            base_id = finger_id
            if finger_id != mp_hands.HandLandmark.THUMB_MCP:
                # Dedos normales
                mcp = hand_points[base_id]
                pip = hand_points[base_id + 1]
                dip = hand_points[base_id + 2]
                tip = hand_points[base_id + 3]
                
                # Ángulos de cada articulación
                pip_angle = calculate_angle(mcp, pip, dip)
                dip_angle = calculate_angle(pip, dip, tip)
                
                features.extend([pip_angle, dip_angle])
            else:
                # Pulgar
                cmc = hand_points[mp_hands.HandLandmark.THUMB_CMC]
                mcp = hand_points[mp_hands.HandLandmark.THUMB_MCP]
                ip = hand_points[mp_hands.HandLandmark.THUMB_IP]
                tip = hand_points[mp_hands.HandLandmark.THUMB_TIP]
                
                mcp_angle = calculate_angle(cmc, mcp, ip)
                ip_angle = calculate_angle(mcp, ip, tip)
                
                features.extend([mcp_angle, ip_angle])
        
        # 4. Indicador de qué mano es (izquierda o derecha)
        hand_side = 1 if handedness.classification[0].label == "Right" else 0
        features.append(hand_side)
        
        return np.array(features)
    except Exception as e:
        st.error(f"Error al extraer características: {str(e)}")
        return None

# Función para procesar video
def process_video(mode="collect"):
    """Procesa video de la cámara web usando OpenCV directamente"""
    # Inicializar cámara
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("No se pudo abrir la cámara. Verifica que esté conectada y disponible.")
        return
    
    # Inicializar MediaPipe Hands
    with mp_hands.Hands(
        max_num_hands=2,  # Detectar hasta 2 manos
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        model_complexity=1  # 0 o 1
    ) as hands:
        
        # Para predicción
        prediction = "N/A"
        if mode == "predict" and os.path.exists(os.path.join(MODELS_PATH, 'svm_model.pkl')):
            model = pickle.load(open(os.path.join(MODELS_PATH, 'svm_model.pkl'), 'rb'))
            scaler = pickle.load(open(os.path.join(MODELS_PATH, 'scaler.pkl'), 'rb'))
            classes = pickle.load(open(os.path.join(MODELS_PATH, 'classes.pkl'), 'rb'))
        
        # Variables para control de tiempo
        last_time = time.time()
        
        # Placeholder para la imagen
        img_placeholder = st.empty()
        
        # Indicador de estado
        status_text = st.empty()
        
        running = True
        while running and st.session_state.start_button:
            # Leer frame
            ret, img = cap.read()
            if not ret:
                st.error("No se pudo leer el frame de la cámara.")
                break
            
            # Voltear horizontalmente la imagen (efecto espejo)
            img = cv2.flip(img, 1)
            
            # Convertir a RGB para MediaPipe
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Procesar imagen
            results = hands.process(img_rgb)
            
            # Visualización
            visual_img = img.copy()
            
            # Si se detectan manos
            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    # Dibujar los landmarks de la mano
                    mp_drawing.draw_landmarks(
                        visual_img,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                    
                    # Extraer características
                    features = extract_hand_features(hand_landmarks, handedness)
                    
                    if features is not None:
                        # MODO RECOLECCIÓN
                        if mode == "collect" and st.session_state.active:
                            current_time = time.time()
                            # Tomar datos cada 0.3 segundos
                            if current_time - last_time > 0.3:
                                # Guardar en estado
                                st.session_state.features_data.append(features.tolist())
                                st.session_state.counter += 1
                                last_time = current_time
                                
                                # Actualizar información de estado
                                status_text.text(f"Muestras recolectadas: {st.session_state.counter}")
                        
                        # MODO PREDICCIÓN
                        elif mode == "predict" and 'model' in locals():
                            current_time = time.time()
                            if current_time - last_time > 0.5:
                                # Predecir
                                features_reshaped = features.reshape(1, -1)
                                scaled_features = scaler.transform(features_reshaped)
                                prediction = model.predict(scaled_features)[0]
                                last_time = current_time
                                
                                # Actualizar información de estado
                                status_text.text(f"Predicción: {prediction}")
                                
                                # Mostrar predicción en la imagen
                                cv2.putText(
                                    visual_img,
                                    f"{handedness.classification[0].label} - {prediction}",
                                    (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    (0, 255, 255),
                                    2
                                )
            
            # Mostrar información en pantalla
            if mode == "collect":
                # Estado
                status = "ACTIVO ✅" if st.session_state.active else "INACTIVO ❌"
                cv2.putText(
                    visual_img,
                    f"Estado: {status}",
                    (20, visual_img.shape[0] - 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0) if st.session_state.active else (0, 0, 255),
                    2
                )
                
                # Etiqueta
                cv2.putText(
                    visual_img,
                    f"Etiqueta: {st.session_state.label}",
                    (20, visual_img.shape[0] - 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 0),
                    2
                )
                
                # Contador
                cv2.putText(
                    visual_img,
                    f"Muestras: {st.session_state.counter}",
                    (20, visual_img.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 165, 255),
                    2
                )
            
            # Convertir a RGB para mostrar en Streamlit
            visual_img_rgb = cv2.cvtColor(visual_img, cv2.COLOR_BGR2RGB)
            
            # Mostrar imagen
            img_placeholder.image(visual_img_rgb, channels="RGB")
            
            # Comprobar si se debe detener
            if not st.session_state.start_button:
                running = False
        
        # Liberar la cámara
        cap.release()

# Función para guardar datos (sin modificaciones)
def save_data(label, features):
    """Guarda características en CSV"""
    if not features or len(features) == 0:
        return False
    
    try:
        # Crear DataFrame
        df = pd.DataFrame(features)
        
        # Añadir etiqueta
        df['label'] = label
        
        # Guardar CSV con ruta absoluta
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(DATA_PATH, f"{label}_{timestamp}.csv")
        
        # Guardar con modo de escritura explícito
        df.to_csv(filename, index=False, mode='w')
        
        # Log para depuración
        print(f"Guardado archivo: {filename}")
        print(f"Contenido: {len(df)} filas, {df.shape[1]} columnas")
        
        return filename
    except Exception as e:
        st.error(f"Error al guardar: {str(e)}")
        return None

# Función para entrenar modelo (sin modificaciones)
def train_model():
    """Entrena modelo SVM con datos guardados"""
    files = [f for f in os.listdir(DATA_PATH) if f.endswith('.csv')]
    
    if not files:
        return "No hay archivos de datos"
    
    try:
        # Cargar datos
        datasets = []
        for file in files:
            file_path = os.path.join(DATA_PATH, file)
            df = pd.read_csv(file_path)
            datasets.append(df)
        
        # Combinar datos
        data = pd.concat(datasets, ignore_index=True)
        
        # Separar características y etiquetas
        X = data.drop('label', axis=1)
        y = data['label']
        
        # Normalizar
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Entrenar SVM
        model = SVC(kernel='rbf', C=10, gamma='scale')
        model.fit(X_scaled, y)
        
        # Guardar modelo con ruta absoluta
        pickle.dump(model, open(os.path.join(MODELS_PATH, 'svm_model.pkl'), 'wb'))
        pickle.dump(scaler, open(os.path.join(MODELS_PATH, 'scaler.pkl'), 'wb'))
        pickle.dump(list(model.classes_), open(os.path.join(MODELS_PATH, 'classes.pkl'), 'wb'))
        
        return f"Modelo entrenado con {len(data)} muestras"
    except Exception as e:
        return f"Error al entrenar: {str(e)}"

# Función para iniciar/detener cámara (sin modificaciones)
def toggle_camera():
    st.session_state.start_button = not st.session_state.start_button

# Interfaz principal
st.sidebar.header("Modo")
app_mode = st.sidebar.radio(
    "Seleccionar modo",
    ["Recolección", "Entrenamiento", "Predicción"]
)

# Mostrar rutas de directorio
st.sidebar.subheader("Rutas")
st.sidebar.info(f"Proyecto: {PROJECT_PATH}\nDatos: {DATA_PATH}\nModelos: {MODELS_PATH}")

# MODO RECOLECCIÓN
if app_mode == "Recolección":
    st.header("Recolección de Datos")
    
    # Formulario de etiqueta
    st.session_state.label = st.text_input("Etiqueta para posición de mano:", st.session_state.label, 
                                          placeholder="Ejemplo: puño, palma_abierta, pulgar_arriba")
    
    # Instrucciones
    st.markdown("""
    ### 📋 Instrucciones
    1. Escribe una etiqueta descriptiva para la posición de mano que quieres capturar
    2. Asegúrate de estar en un lugar con buena iluminación
    3. Coloca la mano a una distancia adecuada para que sea claramente visible
    4. Presiona "Iniciar Cámara" y luego "Iniciar Recolección"
    5. Recolecta al menos 50-100 muestras para cada posición
    
    > 💡 **Consejo**: Para obtener un mejor modelo, varía ligeramente el ángulo y la posición de la mano mientras recopilas datos.
    """)
    
    # Botones de control
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        # Botón para iniciar/detener cámara
        if st.button("Iniciar/Detener Cámara"):
            toggle_camera()
    with col2:
        if st.button("Iniciar Recolección"):
            st.session_state.active = True
    with col3:
        if st.button("Detener Recolección"):
            st.session_state.active = False
    with col4:
        if st.button("Reiniciar Contador"):
            st.session_state.features_data = []
            st.session_state.counter = 0
    
    # Iniciar procesamiento de video si se ha pulsado el botón
    if st.session_state.start_button:
        process_video(mode="collect")
    
    # Información de recolección
    st.info(f"Datos recolectados: {st.session_state.counter} muestras")
    
    # Botón para guardar
    if st.button("Guardar Datos"):
        if st.session_state.features_data and len(st.session_state.features_data) > 0:
            filename = save_data(st.session_state.label, st.session_state.features_data)
            if filename:
                st.success(f"Datos guardados en: {filename}")
                st.session_state.features_data = []
                st.session_state.counter = 0
            else:
                st.error("Error al guardar datos")
        else:
            st.warning("No hay datos para guardar")
    
    # Lista de archivos
    st.subheader("Archivos existentes")
    files = [f for f in os.listdir(DATA_PATH) if f.endswith('.csv')]
    if files:
        for file in files:
            file_path = os.path.join(DATA_PATH, file)
            try:
                df = pd.read_csv(file_path)
                st.text(f"{file}: {len(df)} muestras, Etiqueta: {df['label'].iloc[0]}")
            except Exception as e:
                st.text(f"{file}: Error al leer ({str(e)})")
    else:
        st.text("No hay archivos guardados")

# MODO ENTRENAMIENTO
elif app_mode == "Entrenamiento":
    st.header("Entrenamiento de Modelo SVM")
    
    # Información de datos
    st.subheader("Datos disponibles")
    files = [f for f in os.listdir(DATA_PATH) if f.endswith('.csv')]
    
    if files:
        data_info = {}
        total = 0
        
        for file in files:
            file_path = os.path.join(DATA_PATH, file)
            try:
                df = pd.read_csv(file_path)
                label = df['label'].iloc[0]
                count = len(df)
                total += count
                
                if label in data_info:
                    data_info[label] += count
                else:
                    data_info[label] = count
            except Exception as e:
                st.warning(f"Error al leer {file}: {str(e)}")
        
        # Mostrar resumen
        st.write(f"Total: {total} muestras")
        st.write("Distribución:")
        for label, count in data_info.items():
            st.write(f"- {label}: {count} muestras ({count/total*100:.1f}%)")
        
        # Recomendaciones
        st.info("""
        ### 💡 Recomendaciones para mejores resultados:
        - Asegúrate de tener al menos 50-100 muestras por posición de mano
        - La distribución de clases debe ser equilibrada (similar número de muestras para cada posición)
        - Para mayor precisión, recolecta datos en diferentes condiciones de iluminación y ángulos
        """)
        
        # Botón para entrenar
        if st.button("Entrenar Modelo"):
            with st.spinner("Entrenando..."):
                result = train_model()
                st.success(result)
    else:
        st.warning("No hay datos disponibles")

# MODO PREDICCIÓN
else:
    st.header("Predicción de Posiciones de Manos")
    
    # Comprobar modelo
    model_path = os.path.join(MODELS_PATH, 'svm_model.pkl')
    if os.path.exists(model_path):
        # Cargar clases
        classes = pickle.load(open(os.path.join(MODELS_PATH, 'classes.pkl'), 'rb'))
        st.write(f"Modelo cargado. Posiciones reconocidas: {', '.join(classes)}")
        
        # Botón para iniciar/detener cámara
        if st.button("Iniciar/Detener Cámara"):
            toggle_camera()
            
        # Iniciar procesamiento de video si se ha pulsado el botón
        if st.session_state.start_button:
            process_video(mode="predict")
            
        st.info("Muestra tus manos frente a la cámara para ver la predicción")
        
        # Consejos
        st.markdown("""
        ### 📋 Consejos para la predicción:
        1. Asegúrate de que tus manos sean claramente visibles en la cámara
        2. Mantén buena iluminación para una detección precisa
        3. Intenta replicar las posiciones con las que entrenaste el modelo
        4. Mantén la posición estable durante unos segundos
        5. El sistema puede detectar hasta 2 manos simultáneamente
        """)
    else:
        st.warning(f"No hay modelo entrenado disponible en {model_path}")
        st.info("Primero debes recolectar datos y entrenar un modelo en las secciones anteriores.")

# Información de estado
st.sidebar.subheader("Estado")
st.sidebar.info(f"""
- Modo: {app_mode}
- Etiqueta: {st.session_state.label}
- Muestras: {st.session_state.counter}
- Recolección: {"Activa" if st.session_state.active else "Inactiva"}
- Cámara: {"Activa" if st.session_state.start_button else "Inactiva"}
""")

# Sugerencias de posiciones de manos
st.sidebar.subheader("Ejemplos de Posiciones")
st.sidebar.markdown("""
### Gestos comunes:
- Palma abierta
- Puño cerrado
- Pulgar arriba
- OK (pulgar e índice)
- Paz (V con índice y medio)
- Rock and roll (cuernos)
- Mano plana
- Puntero (dedo índice)
- Pistola
- Números (1-5 dedos)

### Posiciones específicas:
- Señalar hacia abajo
- Garra
- Posición de relajación
- Mano en forma de taza
""")
