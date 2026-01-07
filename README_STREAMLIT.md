# 🏗️ Interface Streamlit - Extraction de Plans Architecturaux

## 🚀 Démarrage Rapide

### Option 1: Script Windows (Recommandé)
Double-cliquez sur `lancer_streamlit.bat` ou exécutez dans PowerShell:
```powershell
.\lancer_streamlit.bat
```

### Option 2: Ligne de commande
```bash
# Activer l'environnement virtuel
venv\Scripts\activate  # Windows
# ou
source venv/bin/activate  # Linux/Mac

# Lancer Streamlit
streamlit run app_streamlit.py
```

## 📋 Prérequis

Assurez-vous que toutes les dépendances sont installées:
```bash
pip install -r requirements.txt
```

Les dépendances principales incluent:
- `streamlit>=1.28.0` - Interface utilisateur
- `ultralytics>=8.0.0` - Modèles YOLO
- `PyMuPDF>=1.23.0` - Extraction PDF
- `opencv-python>=4.8.0` - Traitement d'images
- `pillow>=10.0.0` - Manipulation d'images

## 🎯 Fonctionnalités de l'Interface

### 📄 Onglet "Traitement PDF"
- **Upload de PDF**: Glissez-déposez ou sélectionnez un fichier PDF
- **Configuration**: 
  - Choix du modèle YOLO (yolov8n à yolov8x)
  - Ajustement du seuil de confiance
  - Options avancées (preprocessing, nombre de pages, etc.)
- **Traitement**: Extraction automatique et détection des éléments

### 📊 Onglet "Résultats"
- **Statistiques**: Nombre de pages, détections, confiance moyenne
- **Graphiques**: Répartition des détections par classe
- **Visualisation**: Image annotée avec bounding boxes
- **Tableau détaillé**: Liste complète des détections avec coordonnées
- **Téléchargement**: 
  - Annotations en JSON
  - Image annotée en PNG

### ℹ️ Onglet "À propos"
- Documentation complète
- Instructions d'utilisation
- Technologies utilisées

## 🔧 Configuration

### Modèles YOLO disponibles
- `yolov8n.pt` - Nano (le plus rapide, moins précis)
- `yolov8s.pt` - Small
- `yolov8m.pt` - Medium
- `yolov8l.pt` - Large
- `yolov8x.pt` - Extra Large (le plus précis, plus lent)

**Note**: Les modèles seront téléchargés automatiquement lors de la première utilisation.

### Seuil de confiance
Ajustez le seuil de confiance (0.0 à 1.0) pour filtrer les détections:
- **Faible (0.1-0.3)**: Plus de détections, mais plus de faux positifs
- **Moyen (0.3-0.5)**: Équilibre entre précision et rappel
- **Élevé (0.5-0.9)**: Moins de détections, mais plus précises

## 📖 Utilisation

1. **Lancez l'application** avec `streamlit run app_streamlit.py`
2. **Uploadez un PDF** dans l'onglet "Traitement PDF"
3. **Configurez** les paramètres dans la sidebar si nécessaire
4. **Cliquez sur "Traiter le PDF"** et attendez le traitement
5. **Visualisez les résultats** dans l'onglet "Résultats"
6. **Téléchargez** les annotations si nécessaire

## 🐛 Dépannage

### L'application ne démarre pas
- Vérifiez que Streamlit est installé: `pip install streamlit`
- Vérifiez que l'environnement virtuel est activé
- Vérifiez les erreurs dans la console

### Erreur lors du traitement
- Vérifiez que PyMuPDF est installé: `pip install PyMuPDF`
- Vérifiez que les modèles YOLO peuvent être téléchargés (connexion internet)
- Vérifiez que le PDF n'est pas corrompu

### Détections manquantes
- Réduisez le seuil de confiance
- Essayez un modèle YOLO plus grand (yolov8m, yolov8l)
- Vérifiez que le PDF contient bien des plans architecturaux

### Performance lente
- Utilisez un modèle YOLO plus petit (yolov8n)
- Limitez le nombre de pages à traiter
- Fermez les autres applications

## 📝 Notes

- Le premier traitement peut être lent (téléchargement des modèles)
- Les PDFs volumineux (>50 pages) peuvent prendre du temps
- Les résultats sont sauvegardés dans `data/images/` et `output/`

## 🔗 Liens utiles

- [Documentation Streamlit](https://docs.streamlit.io/)
- [Documentation Ultralytics YOLO](https://docs.ultralytics.com/)
- [Guide d'utilisation complet](GUIDE_UTILISATION.md)

