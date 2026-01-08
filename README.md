# 🏗️ Extraction de Plans Numériques depuis PDF d'Architecte

Projet d'extraction automatique de plans numériques exploitables (format vectoriel ou BIM) à partir de documents PDF d'architecte grâce à la vision par ordinateur et au deep learning.

## 🚀 Démarrage Rapide

### 1. Installation

```bash
# Cloner ou télécharger le projet
cd ArchiProject

# Installer les dépendances
pip install -r requirements.txt
```

### 2. Utilisation

```bash
# Lancer Jupyter
jupyter notebook

# Ouvrir extraction_plans_architecte.ipynb
# Exécuter les cellules dans l'ordre
```

### 3. Code Minimal

```python
# Après avoir exécuté Section 1 (Setup)
pdf_path = "data/pdfs/mon_plan.pdf"
result = process_pdf(pdf_path)  # Extraction
annotations = labeler.label_image(result['saved_paths'][0])  # Détection
labeler.visualize_annotations(result['saved_paths'][0], annotations)  # Visualiser
plt.show()
```

## 📚 Documentation

- **📖 Guide Complet**: [`GUIDE_UTILISATION.md`](GUIDE_UTILISATION.md) - Guide détaillé étape par étape
- **🎯 Prompt Expert**: [`PROMPT_EXPERT.md`](PROMPT_EXPERT.md) - Cahier des charges technique
- **📓 Notebook**: `extraction_plans_architecte.ipynb` - Code complet avec exemples

## 🎯 Fonctionnalités

- ✅ Extraction PDF → Images haute résolution
- ✅ Preprocessing (binarisation, dénoisage, correction)
- ✅ Détection automatique avec YOLO (portes, fenêtres, murs, etc.)
- ✅ Segmentation sémantique
- ✅ Génération de données synthétiques avec LLM
- ✅ Export vectoriel (DXF) et BIM (IFC)
- ✅ Visualisation interactive (Matplotlib, Plotly)
- ✅ Intégration Snowflake (optionnel)

## 📋 Structure du Projet

```
ArchiProject/
├── extraction_plans_architecte.ipynb  # Notebook principal
├── GUIDE_UTILISATION.md               # Guide complet
├── PROMPT_EXPERT.md                   # Cahier des charges
├── requirements.txt                   # Dépendances
├── README.md                          # Ce fichier
├── data/
│   ├── pdfs/                          # Placez vos PDFs ici
│   ├── images/                        # Images extraites
│   ├── annotations/                   # Annotations YOLO
│   └── synthetic_images/             # Images générées
├── models/                            # Modèles entraînés
└── output/                            # Résultats finaux
```

## 🔧 Prérequis

- Python 3.8+
- Jupyter Notebook
- CUDA (optionnel, pour GPU)
- Poppler (pour pdf2image, optionnel)

## 📖 Sections du Notebook

1. **Setup et Imports** ⚠️ OBLIGATOIRE
2. **Snowflake Configuration** (optionnel)
3. **Extraction PDF** ⚠️ OBLIGATOIRE
4. **Génération Données LLM** (optionnel)
5. **Labellisation** ⚠️ OBLIGATOIRE
6. **Entraînement YOLO** (optionnel, long)
7. **Inférence** ⚠️ OBLIGATOIRE
8. **Visualisation** ⚠️ OBLIGATOIRE
9. **Tests et Validation** (recommandé)
10. **Exemples d'Utilisation** - Exemples prêts à l'emploi

## 🎓 Utilisation pour l'Évaluation

### Critères d'Évaluation

1. **Qualité du code (25%)**: Code lisible, bien structuré, commenté
2. **Maîtrise Snowflake (25%)**: Connexion, requêtes, stockage
3. **Front-end et Visualisation (25%)**: Dashboard interactif, métriques
4. **Use case et Exécution (25%)**: Pipeline complet, résultats exploitables

### Livrables

- ✅ **Notebook .ipynb** (date limite: 8/01/26)
- ✅ **Présentation orale** (20 min, 09/01/2026)

## 🆘 Support

Voir la section **Dépannage** dans [`GUIDE_UTILISATION.md`](GUIDE_UTILISATION.md)

## 📝 Licence

Projet académique - Utilisation libre pour l'évaluation

---


