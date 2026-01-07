# 📖 Guide d'Utilisation - Extraction de Plans Numériques

## 🚀 Guide Complet pour Exécuter le Projet

Ce guide vous explique étape par étape comment utiliser le notebook `extraction_plans_architecte.ipynb` pour extraire des plans numériques depuis des PDFs d'architecte.

---

## 📋 Table des Matières

1. [Prérequis et Installation](#prérequis-et-installation)
2. [Configuration Initiale](#configuration-initiale)
3. [Exécution du Notebook](#exécution-du-notebook)
4. [Workflow Complet](#workflow-complet)
5. [Détection et Extraction](#détection-et-extraction)
6. [Visualisation des Résultats](#visualisation-des-résultats)
7. [Dépannage](#dépannage)

---

## 1. Prérequis et Installation

### 1.1 Environnement Python

```bash
# Python 3.8 ou supérieur requis
python --version

# Créer un environnement virtuel (recommandé)
python -m venv venv

# Activer l'environnement
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 1.2 Installation des Dépendances

```bash
# Installer toutes les dépendances
pip install -r requirements.txt

# OU installer manuellement:
pip install opencv-python pillow scikit-image ultralytics torch torchvision
pip install PyMuPDF pdf2image pdfplumber
pip install openai anthropic transformers langchain
pip install diffusers controlnet-aux
pip install snowflake-connector-python snowflake-sqlalchemy
pip install matplotlib plotly seaborn
pip install ezdxf ifcopenshell
pip install pandas numpy scipy ipywidgets
```

### 1.3 Installation de Jupyter

```bash
pip install jupyter notebook
# OU
pip install jupyterlab
```

### 1.4 Téléchargement des Modèles YOLO

Les modèles YOLO seront téléchargés automatiquement au premier usage, mais vous pouvez les pré-télécharger.

**Méthode 1: Dans le Notebook (RECOMMANDÉ)**
- Exécuter la cellule **1.4** du notebook qui télécharge automatiquement les modèles

**Méthode 2: Script Python séparé**
```bash
# Exécuter le script fourni
python download_models.py
```

**Méthode 3: PowerShell/CMD (Windows)**
```powershell
# Activer l'environnement virtuel d'abord
venv\Scripts\activate

# Puis exécuter Python directement
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt'); YOLO('yolo11n-seg.pt'); YOLO('yolo11n-pose.pt')"
```

**Méthode 4: Terminal Linux/Mac**
```bash
# Activer l'environnement virtuel
source venv/bin/activate

# Exécuter
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt'); YOLO('yolo11n-seg.pt'); YOLO('yolo11n-pose.pt')"
```

**Note**: Si vous ne téléchargez pas les modèles maintenant, ils le seront automatiquement lors de leur première utilisation dans la Section 5 (Labellisation).

---

## 2. Configuration Initiale

### 2.1 Variables d'Environnement (Optionnel)

Créez un fichier `.env` ou configurez les variables d'environnement:

```bash
# Snowflake (optionnel)
export SNOWFLAKE_ACCOUNT="votre_compte"
export SNOWFLAKE_USER="votre_utilisateur"
export SNOWFLAKE_PASSWORD="votre_mot_de_passe"
export SNOWFLAKE_WAREHOUSE="COMPUTE_WH"
export SNOWFLAKE_DATABASE="ARCHITECTURE_DB"
export SNOWFLAKE_SCHEMA="PLANS_SCHEMA"

# OpenAI (pour génération de données)
export OPENAI_API_KEY="votre_clé_api"
```

### 2.2 Structure des Répertoires

Le notebook créera automatiquement cette structure:

```
ArchiProject/
├── data/
│   ├── pdfs/              # Placez vos PDFs ici
│   ├── images/            # Images extraites
│   ├── annotations/       # Annotations YOLO
│   ├── synthetic_images/   # Images générées
│   └── scenarios_llm.json # Scénarios générés
├── models/                # Modèles entraînés
├── output/                # Résultats finaux
└── extraction_plans_architecte.ipynb
```

### 2.3 Préparer vos Données

1. **Placer vos PDFs** dans `data/pdfs/`
   ```bash
   # Exemple
   cp vos_plans.pdf data/pdfs/
   ```

2. **Vérifier la structure**
   ```python
   from pathlib import Path
   pdf_dir = Path("data/pdfs")
   pdfs = list(pdf_dir.glob("*.pdf"))
   print(f"Nombre de PDFs trouvés: {len(pdfs)}")
   ```

---

## 3. Exécution du Notebook

### 3.1 Lancer Jupyter

```bash
# Depuis le répertoire du projet
jupyter notebook

# OU avec JupyterLab
jupyter lab
```

### 3.2 Ouvrir le Notebook

1. Ouvrir `extraction_plans_architecte.ipynb`
2. Exécuter les cellules **dans l'ordre** (Kernel → Restart & Run All)

### 3.3 Exécution Séquentielle

**IMPORTANT**: Exécutez les sections dans l'ordre:

1. ✅ **Section 1**: Setup et Imports (obligatoire)
2. ✅ **Section 2**: Snowflake Configuration (optionnel)
3. ✅ **Section 3**: Extraction PDF (obligatoire)
4. ⚠️ **Section 4**: Génération Données LLM (optionnel, long)
5. ✅ **Section 5**: Labellisation (obligatoire)
6. ⚠️ **Section 6**: Entraînement YOLO (optionnel, très long)
7. ✅ **Section 7**: Inférence (obligatoire)
8. ✅ **Section 8**: Visualisation (obligatoire)
9. ✅ **Section 9**: Tests (recommandé)

---

## 4. Workflow Complet

### 4.1 Workflow Minimal (Sans Entraînement)

```python
# 1. Setup (Section 1)
# ✅ Exécuter toutes les cellules de la Section 1

# 2. Configuration Snowflake (Section 2)
# ⚠️ Optionnel - peut être ignoré si pas de Snowflake

# 3. Extraction PDF (Section 3)
pdf_path = "data/pdfs/mon_plan.pdf"
result = process_pdf(pdf_path)
print(f"✅ {len(result['images'])} pages extraites")

# 4. Preprocessing (Section 3)
image_path = result['saved_paths'][0]
processed_img = preprocess_image(image_path)

# 5. Labellisation avec YOLO pré-entraîné (Section 5)
annotations = labeler.label_image(image_path)
print(f"✅ {len(annotations['detections'])} détections trouvées")

# 6. Visualisation (Section 8)
fig = labeler.visualize_annotations(image_path, annotations)
plt.show()

# 7. Extraction vectorielle (Section 7)
# Voir section détaillée ci-dessous
```

### 4.2 Workflow Complet (Avec Entraînement)

```python
# 1-3. Setup, Snowflake, Extraction (comme ci-dessus)

# 4. Génération de données synthétiques (Section 4)
scenarios = llm_generator.generate_scenarios_openai()
generated_images = image_generator.generate_images_from_scenarios(
    scenarios[:10],  # Limiter pour test
    config.DATA_DIR / "synthetic_images"
)

# 5. Augmentation de données (Section 4)
augmented = augmenter.augment_dataset(
    result['saved_paths'],
    config.DATA_DIR / "augmented"
)

# 6. Préparation dataset (Section 6)
yolo_dir = Path("data/yolo_dataset")
dataset_prep.create_yolo_structure(yolo_dir)
splits = dataset_prep.split_dataset(
    result['saved_paths'] + augmented,
    train_ratio=0.7, val_ratio=0.2, test_ratio=0.1
)
dataset_prep.copy_to_yolo_structure(splits, yolo_dir, config.ANNOTATIONS_DIR)
config_path = dataset_prep.create_yolo_config(yolo_dir, len(config.CLASSES))

# 7. Entraînement YOLO (Section 6)
# Voir section détaillée ci-dessous

# 8. Inférence avec modèle entraîné (Section 7)
# Voir section détaillée ci-dessous
```

---

## 5. Détection et Extraction

### 5.1 Détection Simple (YOLO Pré-entraîné)

```python
# Charger une image
image_path = "data/images/mon_plan_page_001.png"

# Détection
annotations = labeler.label_image(image_path)

# Afficher les résultats
print(f"Nombre de détections: {len(annotations['detections'])}")
for det in annotations['detections']:
    print(f"  - {det['class_name']}: confiance {det['confidence']:.2f}")

# Visualiser
fig = labeler.visualize_annotations(image_path, annotations)
plt.show()
```

### 5.2 Détection Complète (Détection + Segmentation + Keypoints)

```python
# Détection
detections = labeler.predict_detection(image_path)

# Segmentation
segmentations = labeler.predict_segmentation(image_path)

# Keypoints
keypoints = labeler.predict_keypoints(image_path)

# Combiner
full_annotations = {
    'detections': detections,
    'segmentations': segmentations,
    'keypoints': keypoints
}

# Sauvegarder
save_annotations_snowflake("plan_001", full_annotations)
```

### 5.3 Traitement d'un PDF Complet

```python
def process_complete_pdf(pdf_path: str):
    """Traite un PDF complet de bout en bout"""
    
    # 1. Extraction
    print("📄 Extraction des pages...")
    result = process_pdf(pdf_path)
    
    if not result:
        print("❌ Erreur lors de l'extraction")
        return None
    
    # 2. Preprocessing
    print("🔧 Preprocessing...")
    processed_images = []
    for img_path in result['saved_paths']:
        processed = preprocess_image(img_path)
        processed_path = config.IMAGES_DIR / f"processed_{Path(img_path).name}"
        processed.save(processed_path)
        processed_images.append(str(processed_path))
    
    # 3. Détection sur chaque page
    print("🔍 Détection des éléments...")
    all_annotations = []
    for img_path in processed_images:
        annotations = labeler.label_image(img_path)
        all_annotations.append(annotations)
        print(f"  ✓ {len(annotations['detections'])} détections sur {Path(img_path).name}")
    
    # 4. Résumé
    total_detections = sum(len(ann['detections']) for ann in all_annotations)
    print(f"\n✅ Traitement terminé: {total_detections} détections au total")
    
    return {
        'plan_data': result['plan_data'],
        'annotations': all_annotations,
        'images': processed_images
    }

# Utilisation
result = process_complete_pdf("data/pdfs/mon_plan.pdf")
```

### 5.4 Extraction Vectorielle (DXF/IFC)

```python
# Voir Section 7 du notebook pour le code complet
# Exemple simplifié:

from export_vectoriel import VectorExporter

exporter = VectorExporter(config)

# Exporter en DXF
dxf_path = exporter.export_to_dxf(
    image_path="data/images/plan.png",
    annotations=annotations,
    output_path="output/plan.dxf"
)

# Exporter en IFC (BIM)
ifc_path = exporter.export_to_ifc(
    image_path="data/images/plan.png",
    annotations=annotations,
    output_path="output/plan.ifc"
)
```

---

## 6. Visualisation des Résultats

### 6.1 Visualisation Simple

```python
# Visualiser les annotations
image_path = "data/images/mon_plan_page_001.png"
annotations = labeler.label_image(image_path)

fig = labeler.visualize_annotations(image_path, annotations)
plt.show()
```

### 6.2 Dashboard Interactif (Plotly)

```python
# Voir Section 8 du notebook
# Le notebook contient un dashboard interactif complet

from dashboard import create_dashboard

# Créer le dashboard
dashboard = create_dashboard(annotations, image_path)
dashboard.show()
```

### 6.3 Comparaison Avant/Après

```python
# Charger image originale
original = Image.open(image_path)

# Charger image avec annotations
annotated = labeler.visualize_annotations(image_path, annotations)

# Comparaison côte à côte
fig, axes = plt.subplots(1, 2, figsize=(20, 10))
axes[0].imshow(original)
axes[0].set_title("Original")
axes[0].axis('off')

axes[1].imshow(annotated)
axes[1].set_title("Avec Détections")
axes[1].axis('off')

plt.tight_layout()
plt.show()
```

### 6.4 Statistiques et Métriques

```python
# Analyser les détections
import pandas as pd

# Créer un DataFrame
detections_data = []
for ann in all_annotations:
    for det in ann['detections']:
        detections_data.append({
            'classe': det['class_name'],
            'confidence': det['confidence'],
            'page': ann.get('page', 0)
        })

df = pd.DataFrame(detections_data)

# Statistiques
print("📊 Statistiques des détections:")
print(df.groupby('classe').agg({
    'confidence': ['mean', 'count']
}))

# Graphique
import seaborn as sns
sns.countplot(data=df, x='classe')
plt.xticks(rotation=45)
plt.title("Distribution des classes détectées")
plt.show()
```

---

## 7. Entraînement YOLO (Optionnel)

### 7.1 Préparation du Dataset

```python
# 1. Collecter toutes les images
all_images = list(config.IMAGES_DIR.glob("*.png"))
all_images += list((config.DATA_DIR / "synthetic_images").glob("*.png"))
all_images += list((config.DATA_DIR / "augmented").glob("*.png"))

# 2. Créer structure YOLO
yolo_dir = Path("data/yolo_dataset")
dataset_prep.create_yolo_structure(yolo_dir)

# 3. Split train/val/test
splits = dataset_prep.split_dataset(
    [str(p) for p in all_images],
    train_ratio=0.7, val_ratio=0.2, test_ratio=0.1
)

# 4. Copier dans structure YOLO
dataset_prep.copy_to_yolo_structure(
    splits, 
    yolo_dir, 
    config.ANNOTATIONS_DIR
)

# 5. Créer config
config_path = dataset_prep.create_yolo_config(
    yolo_dir, 
    len(config.CLASSES)
)
```

### 7.2 Entraînement

```python
from ultralytics import YOLO

# Charger modèle pré-entraîné
model = YOLO("yolo11n.pt")

# Entraîner
results = model.train(
    data=str(config_path),  # Chemin vers data.yaml
    epochs=100,
    imgsz=640,
    batch=16,
    name="plans_architecte",
    project="models"
)

# Le modèle sera sauvegardé dans models/plans_architecte/
```

### 7.3 Évaluation

```python
# Évaluer le modèle
metrics = model.val()

print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")
print(f"Precision: {metrics.box.mp}")
print(f"Recall: {metrics.box.mr}")
```

### 7.4 Utiliser le Modèle Entraîné

```python
# Charger le modèle entraîné
trained_model = YOLO("models/plans_architecte/weights/best.pt")

# Utiliser pour détection
results = trained_model.predict(
    "data/images/nouveau_plan.png",
    conf=0.25
)

# Visualiser
results[0].show()
```

---

## 8. Interface Streamlit (Optionnel)

### 8.1 Créer l'Interface

Créez un fichier `app_streamlit.py`:

```python
import streamlit as st
from PIL import Image
import sys
sys.path.append('.')

# Importer les fonctions du notebook
from extraction_plans_architecte import (
    process_pdf, labeler, config
)

st.title("🏗️ Extraction de Plans Numériques")

# Upload PDF
uploaded_file = st.file_uploader("Télécharger un PDF", type="pdf")

if uploaded_file:
    # Sauvegarder temporairement
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Traiter
    if st.button("Extraire"):
        with st.spinner("Traitement en cours..."):
            result = process_pdf("temp.pdf")
            
            if result:
                st.success(f"✅ {len(result['images'])} pages extraites")
                
                # Afficher première page
                st.image(result['images'][0], caption="Page 1")
                
                # Détection
                annotations = labeler.label_image(result['saved_paths'][0])
                st.write(f"🔍 {len(annotations['detections'])} détections")
                
                # Afficher détections
                for det in annotations['detections']:
                    st.write(f"- {det['class_name']}: {det['confidence']:.2%}")
```

### 8.2 Lancer Streamlit

```bash
streamlit run app_streamlit.py
```

---

## 9. Dépannage

### 9.1 Erreurs Communes

**Erreur: "Module not found"**
```bash
pip install <nom_module>
```

**Erreur: "CUDA out of memory"**
```python
# Réduire la taille du batch
config.YOLO_IMG_SIZE = 416  # Au lieu de 640
```

**Erreur: "PDF extraction failed"**
```bash
# Installer poppler (pour pdf2image)
# Windows: télécharger depuis https://github.com/oschwartz10612/poppler-windows
# Linux: sudo apt-get install poppler-utils
# Mac: brew install poppler
```

**Erreur: "Snowflake connection failed"**
```python
# Le notebook fonctionne en mode simulation sans Snowflake
# Vérifier les variables d'environnement si nécessaire
```

### 9.2 Vérifications

```python
# Vérifier les imports
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA disponible: {torch.cuda.is_available()}")

# Vérifier YOLO
from ultralytics import YOLO
model = YOLO("yolo11n.pt")
print("✅ YOLO fonctionne")

# Vérifier les chemins
from pathlib import Path
print(f"PDFs: {list(Path('data/pdfs').glob('*.pdf'))}")
print(f"Images: {list(Path('data/images').glob('*.png'))}")
```

---

## 10. Exemples d'Utilisation Rapide

### 10.1 Exemple Minimal

```python
# 1. Setup
# Exécuter Section 1 du notebook

# 2. Traiter un PDF
pdf_path = "data/pdfs/mon_plan.pdf"
result = process_pdf(pdf_path)

# 3. Détecter
annotations = labeler.label_image(result['saved_paths'][0])

# 4. Visualiser
labeler.visualize_annotations(result['saved_paths'][0], annotations)
plt.show()
```

### 10.2 Exemple Complet

```python
# Voir le notebook complet pour l'exemple détaillé
# Toutes les sections sont documentées avec des exemples
```

---

## 📞 Support

Pour toute question ou problème:
1. Vérifier la section Dépannage
2. Consulter la documentation dans le notebook
3. Vérifier les logs d'erreur

---

**Bon travail ! 🚀**

