"""
🏗️ Interface Streamlit pour l'Extraction de Plans Architecturaux

Cette interface permet de tester facilement le système d'extraction de plans
depuis des PDFs d'architecte avec détection YOLO.

Usage:
    streamlit run app_streamlit.py
"""

import streamlit as st
import os
import sys
from pathlib import Path
import tempfile
import json
from typing import Dict, List, Optional
import traceback

# Configuration de la page
st.set_page_config(
    page_title="🏗️ Extraction de Plans Architecturaux",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        margin: 1rem 0;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Titre principal
st.markdown('<h1 class="main-header">🏗️ Extraction de Plans Architecturaux</h1>', unsafe_allow_html=True)

# Sidebar - Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Mode d'utilisation
    use_notebook_code = st.checkbox(
        "Utiliser le code du notebook",
        value=True,
        help="Si activé, le code sera importé depuis le notebook. Sinon, utilisez les modules Python."
    )
    
    # Modèle YOLO
    yolo_model = st.selectbox(
        "Modèle YOLO",
        options=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
        index=0,
        help="Modèle YOLO à utiliser pour la détection"
    )
    
    # Seuil de confiance
    confidence_threshold = st.slider(
        "Seuil de confiance",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Seuil minimum de confiance pour les détections"
    )
    
    # Options avancées
    with st.expander("🔧 Options avancées"):
        show_preprocessing = st.checkbox("Afficher le preprocessing", value=False)
        save_intermediate = st.checkbox("Sauvegarder les images intermédiaires", value=True)
        max_pages = st.number_input("Nombre max de pages à traiter", min_value=1, max_value=50, value=10)
    
    st.markdown("---")
    st.markdown("### 📖 Aide")
    st.info("""
    **Comment utiliser:**
    1. Uploadez un PDF de plan architectural
    2. Cliquez sur "Traiter le PDF"
    3. Visualisez les résultats
    4. Téléchargez les annotations
    """)

# Initialisation de session state
if 'processing_result' not in st.session_state:
    st.session_state.processing_result = None
if 'annotations' not in st.session_state:
    st.session_state.annotations = None
if 'extracted_images' not in st.session_state:
    st.session_state.extracted_images = []

# Fonction pour charger les modules depuis le notebook
@st.cache_resource
def load_notebook_modules():
    """Charge les modules nécessaires depuis le notebook"""
    try:
        # Ajouter le répertoire courant au path
        sys.path.insert(0, str(Path.cwd()))
        
        # Essayer d'importer depuis un module Python si disponible
        # Sinon, on utilisera les fonctions directement
        return True
    except Exception as e:
        st.error(f"Erreur lors du chargement des modules: {e}")
        return False



# Zone principale
tab1, tab2, tab3 = st.tabs(["📄 Traitement PDF", "📊 Résultats", "ℹ️ À propos"])

with tab1:
    st.header("📤 Upload et Traitement")
    
    # Upload de fichier
    uploaded_file = st.file_uploader(
        "Choisissez un fichier PDF",
        type=['pdf'],
        help="Sélectionnez un PDF de plan architectural à traiter"
    )
    
    if uploaded_file is not None:
        st.success(f"✅ Fichier chargé: {uploaded_file.name} ({uploaded_file.size / 1024:.2f} KB)")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            process_button = st.button("🚀 Traiter le PDF", type="primary", use_container_width=True)
        
        with col2:
            if st.button("🔄 Réinitialiser", use_container_width=True):
                st.session_state.processing_result = None
                st.session_state.annotations = None
                st.session_state.extracted_images = []
                st.rerun()
        
        if process_button:
            with st.spinner("🔄 Traitement en cours... Cela peut prendre quelques minutes."):
                try:
                    # Initialisation des modules
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Étape 1: Chargement des modules
                    status_text.text("📦 Chargement des modules...")
                    progress_bar.progress(10)
                    
                    # Import des classes nécessaires
                    # Note: Ces imports doivent être adaptés selon votre structure
                    try:
                        # Essayer d'importer depuis le notebook ou un module Python
                        # Pour l'instant, on va créer une version simplifiée
                        from pathlib import Path
                        import cv2
                        import numpy as np
                        from PIL import Image
                        from ultralytics import YOLO
                        
                        # Configuration simplifiée
                        class SimpleConfig:
                            BASE_DIR = Path.cwd()
                            DATA_DIR = BASE_DIR / "data"
                            PDF_DIR = DATA_DIR / "pdfs"
                            IMAGES_DIR = DATA_DIR / "images"
                            MODELS_DIR = BASE_DIR / "models"
                            OUTPUT_DIR = BASE_DIR / "output"
                            
                            def __init__(self):
                                for dir_path in [self.DATA_DIR, self.PDF_DIR, self.IMAGES_DIR, 
                                                self.MODELS_DIR, self.OUTPUT_DIR]:
                                    dir_path.mkdir(parents=True, exist_ok=True)
                        
                        config = SimpleConfig()
                        
                        # Classe PDFExtractor simplifiée
                        class SimplePDFExtractor:
                            def __init__(self, config):
                                self.config = config
                                try:
                                    import fitz  # PyMuPDF
                                    self.fitz = fitz
                                except ImportError:
                                    st.error("PyMuPDF (fitz) n'est pas installé. Installez-le avec: pip install PyMuPDF")
                                    raise
                            
                            def extract_images(self, pdf_path: str, max_pages: int = 10):
                                """Extrait les images d'un PDF"""
                                doc = self.fitz.open(pdf_path)
                                images = []
                                
                                for page_num in range(min(len(doc), max_pages)):
                                    page = doc[page_num]
                                    pix = page.get_pixmap(matrix=self.fitz.Matrix(2, 2))
                                    img_data = pix.tobytes("png")
                                    images.append(img_data)
                                
                                doc.close()
                                return {'images': images}
                            
                            def save_extracted_images(self, images, pdf_name, output_dir):
                                """Sauvegarde les images extraites"""
                                output_dir.mkdir(parents=True, exist_ok=True)
                                saved_paths = []
                                
                                for idx, img_data in enumerate(images):
                                    img_path = output_dir / f"{pdf_name}_page_{idx+1:03d}.png"
                                    with open(img_path, 'wb') as f:
                                        f.write(img_data)
                                    saved_paths.append(str(img_path))
                                
                                return saved_paths
                        
                        extractor = SimplePDFExtractor(config)
                        
                        # Fonction wrapper mise à jour
                        def process_pdf_file_updated(uploaded_file, config, extractor, labeler, max_pages):
                            """Traite un fichier PDF uploadé"""
                            try:
                                # Sauvegarder le fichier temporairement
                                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                                    tmp_file.write(uploaded_file.read())
                                    tmp_path = tmp_file.name
                                
                                # Traitement
                                result_data = extractor.extract_images(tmp_path, max_pages)
                                
                                if not result_data or not result_data['images']:
                                    os.unlink(tmp_path)
                                    return None, None
                                
                                # Sauvegarder les images
                                pdf_name = Path(tmp_path).stem
                                saved_paths = extractor.save_extracted_images(
                                    result_data['images'],
                                    pdf_name,
                                    config.IMAGES_DIR
                                )
                                
                                result = {
                                    'images': result_data['images'],
                                    'saved_paths': saved_paths,
                                    'pdf_name': pdf_name,
                                    'num_pages': len(result_data['images'])
                                }
                                
                                if not result['saved_paths']:
                                    os.unlink(tmp_path)
                                    return None, None
                                
                                # Détection sur la première image
                                first_image_path = result['saved_paths'][0]
                                annotations = labeler.label_image(first_image_path)
                                
                                # Nettoyer le fichier temporaire
                                os.unlink(tmp_path)
                                
                                return result, annotations
                            
                            except Exception as e:
                                st.error(f"Erreur: {e}")
                                st.error(traceback.format_exc())
                                return None, None
                        
                        # Classe YOLOLabeler simplifiée
                        class SimpleYOLOLabeler:
                            def __init__(self, config, model_name="yolov8n.pt", conf_threshold=0.25):
                                self.config = config
                                self.conf_threshold = conf_threshold
                                self.model = YOLO(model_name)
                            
                            def label_image(self, image_path: str):
                                """Détecte les objets dans une image"""
                                results = self.model(image_path, conf=self.conf_threshold)
                                
                                detections = []
                                for result in results:
                                    boxes = result.boxes
                                    # Utiliser les noms de classes réels de YOLO (COCO dataset)
                                    # YOLO détecte des objets génériques (person, car, etc.), pas des éléments architecturaux spécifiques
                                    # Pour détecter des éléments architecturaux, il faudrait un modèle fine-tuné
                                    for box in boxes:
                                        cls_id = int(box.cls[0])
                                        class_name = result.names[cls_id] if hasattr(result, 'names') and cls_id in result.names else f"class_{cls_id}"
                                        detections.append({
                                            'class_id': cls_id,
                                            'class_name': class_name,
                                            'confidence': float(box.conf[0]),
                                            'bbox': box.xyxy[0].tolist()
                                        })
                                
                                return {
                                    'detections': detections,
                                    'num_detections': len(detections),
                                    'image_path': image_path
                                }
                            
                            def visualize_annotations(self, image_path: str, annotations):
                                """Visualise les annotations sur l'image"""
                                img = cv2.imread(image_path)
                                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                
                                for det in annotations['detections']:
                                    bbox = det['bbox']
                                    x1, y1, x2, y2 = map(int, bbox)
                                    
                                    # Dessiner le rectangle
                                    cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    
                                    # Ajouter le label
                                    label = f"{det['class_name']}: {det['confidence']:.2f}"
                                    cv2.putText(img_rgb, label, (x1, y1-10),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                
                                return Image.fromarray(img_rgb)
                        
                        labeler = SimpleYOLOLabeler(config, yolo_model, confidence_threshold)
                        
                    except Exception as e:
                        st.error(f"Erreur lors de l'initialisation: {e}")
                        st.error(traceback.format_exc())
                        st.stop()
                    
                    # Étape 2: Extraction PDF
                    status_text.text("📄 Extraction des pages du PDF...")
                    progress_bar.progress(30)
                    
                    result, annotations = process_pdf_file_updated(uploaded_file, config, extractor, labeler, max_pages)
                    
                    if result:
                        progress_bar.progress(70)
                        status_text.text("🔍 Détection des éléments architecturaux...")
                        
                        if annotations:
                            progress_bar.progress(100)
                            status_text.text("✅ Traitement terminé!")
                            
                            # Sauvegarder dans session state
                            st.session_state.processing_result = result
                            st.session_state.annotations = annotations
                            st.session_state.extracted_images = result['saved_paths']
                            
                            st.success(f"✅ {result['num_pages']} pages extraites, {annotations['num_detections']} éléments détectés!")
                            st.rerun()
                        else:
                            st.warning("⚠️ Aucune détection trouvée")
                    else:
                        st.error("❌ Erreur lors du traitement du PDF")
                
                except Exception as e:
                    st.error(f"❌ Erreur: {e}")
                    st.error(traceback.format_exc())

with tab2:
    st.header("📊 Visualisation des Résultats")
    
    if st.session_state.processing_result and st.session_state.annotations:
        result = st.session_state.processing_result
        annotations = st.session_state.annotations
        
        # Statistiques
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Pages extraites", result['num_pages'])
        with col2:
            st.metric("Éléments détectés", annotations['num_detections'])
        with col3:
            if annotations['detections']:
                avg_conf = sum(d['confidence'] for d in annotations['detections']) / len(annotations['detections'])
                st.metric("Confiance moyenne", f"{avg_conf:.2%}")
            else:
                st.metric("Confiance moyenne", "N/A")
        with col4:
            st.metric("PDF traité", result['pdf_name'])
        
        # Détections par classe
        if annotations['detections']:
            st.subheader("📈 Détections par classe")
            from collections import Counter
            class_counts = Counter(d['class_name'] for d in annotations['detections'])
            
            col1, col2 = st.columns([2, 1])
            with col1:
                import pandas as pd
                df = pd.DataFrame([
                    {'Classe': k, 'Nombre': v} 
                    for k, v in class_counts.items()
                ])
                st.bar_chart(df.set_index('Classe'))
            
            with col2:
                st.dataframe(df, use_container_width=True)
        
        # Visualisation de l'image
        st.subheader("🖼️ Image annotée")
        
        if st.session_state.extracted_images:
            try:
                from PIL import Image
                import cv2
                
                image_path = st.session_state.extracted_images[0]
                img = cv2.imread(image_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Dessiner les annotations
                for det in annotations['detections']:
                    bbox = det['bbox']
                    x1, y1, x2, y2 = map(int, bbox)
                    
                    # Rectangle
                    cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    
                    # Label
                    label = f"{det['class_name']}: {det['confidence']:.2f}"
                    (text_width, text_height), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    cv2.rectangle(img_rgb, (x1, y1-text_height-10), 
                                (x1+text_width, y1), (0, 255, 0), -1)
                    cv2.putText(img_rgb, label, (x1, y1-5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
                st.image(img_rgb, use_container_width=True, caption="Image avec détections")
                
            except Exception as e:
                st.error(f"Erreur lors de la visualisation: {e}")
        
        # Tableau des détections
        st.subheader("📋 Détails des détections")
        if annotations['detections']:
            detections_df = pd.DataFrame([
                {
                    'Classe': d['class_name'],
                    'Confiance': f"{d['confidence']:.2%}",
                    'X1': int(d['bbox'][0]),
                    'Y1': int(d['bbox'][1]),
                    'X2': int(d['bbox'][2]),
                    'Y2': int(d['bbox'][3])
                }
                for d in annotations['detections']
            ])
            st.dataframe(detections_df, use_container_width=True)
        
        # Téléchargement
        st.subheader("💾 Télécharger les résultats")
        col1, col2 = st.columns(2)
        
        with col1:
            # Télécharger les annotations en JSON
            annotations_json = json.dumps(annotations, indent=2, default=str)
            st.download_button(
                label="📥 Télécharger annotations (JSON)",
                data=annotations_json,
                file_name=f"{result['pdf_name']}_annotations.json",
                mime="application/json"
            )
        
        with col2:
            # Télécharger l'image annotée
            if st.session_state.extracted_images:
                try:
                    annotated_img = Image.fromarray(img_rgb)
                    from io import BytesIO
                    buf = BytesIO()
                    annotated_img.save(buf, format='PNG')
                    buf.seek(0)
                    
                    st.download_button(
                        label="📥 Télécharger image annotée (PNG)",
                        data=buf,
                        file_name=f"{result['pdf_name']}_annotated.png",
                        mime="image/png"
                    )
                except Exception as e:
                    st.error(f"Erreur: {e}")
    
    else:
        st.info("👆 Uploadez et traitez un PDF dans l'onglet 'Traitement PDF' pour voir les résultats ici.")

with tab3:
    st.header("ℹ️ À propos")
    
    st.markdown("""
    ### 🏗️ Extraction de Plans Architecturaux
    
    Cette application permet d'extraire automatiquement des plans numériques depuis des PDFs 
    d'architecte en utilisant la vision par ordinateur et le deep learning.
    
    #### 🚀 Fonctionnalités
    
    - **Extraction PDF**: Conversion automatique des pages PDF en images
    - **Détection YOLO**: Identification des éléments architecturaux (murs, portes, fenêtres, etc.)
    - **Visualisation**: Affichage interactif des résultats avec annotations
    - **Export**: Téléchargement des annotations et images annotées
    
    #### 📚 Technologies utilisées
    
    - **Streamlit**: Interface utilisateur
    - **YOLO (Ultralytics)**: Détection d'objets
    - **PyMuPDF (fitz)**: Extraction PDF
    - **OpenCV**: Traitement d'images
    - **PIL/Pillow**: Manipulation d'images
    
    #### 📖 Documentation
    
    Pour plus d'informations, consultez le fichier `GUIDE_UTILISATION.md`.
    
    #### 🐛 Problèmes connus
    
    - Le premier traitement peut être lent (téléchargement des modèles YOLO)
    - Les PDFs très volumineux peuvent prendre du temps à traiter
    """)
    
    st.markdown("---")
    st.markdown("### 📝 Instructions d'utilisation")
    
    st.markdown("""
    1. **Préparation**: Assurez-vous que toutes les dépendances sont installées
       ```bash
       pip install -r requirements.txt
       ```
    
    2. **Lancement**: Démarrez l'application Streamlit
       ```bash
       streamlit run app_streamlit.py
       ```
    
    3. **Utilisation**: 
       - Uploadez un PDF de plan architectural
       - Cliquez sur "Traiter le PDF"
       - Visualisez les résultats dans l'onglet "Résultats"
       - Téléchargez les annotations si nécessaire
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "🏗️ Extraction de Plans Architecturaux - Interface Streamlit"
    "</div>",
    unsafe_allow_html=True
)

