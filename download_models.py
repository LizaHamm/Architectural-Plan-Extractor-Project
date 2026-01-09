"""
Script pour télécharger les modèles YOLO
Exécuter avec: python download_models.py
"""

from ultralytics import YOLO
from pathlib import Path



custom_model_path = Path("models/best.pt")

if custom_model_path.exists():
    print("\n Modèle floor plan détecté :", custom_model_path.resolve())
    try:
        model_fp = YOLO(str(custom_model_path))
        print("    Modèle floor plan chargé avec succès")
        print("    Classes :", model_fp.names)
    except Exception as e:
        print("    Erreur chargement modèle floor plan :", e)
else:
    print("\n  Modèle floor plan non trouvé :", custom_model_path.resolve())



print("=" * 60)
print(" TÉLÉCHARGEMENT DES MODÈLES YOLO")
print("=" * 60)
print("\n Cela peut prendre quelques minutes lors du premier téléchargement...\n")

try:
    # Modèle détection (le plus utilisé)
    print(" Téléchargement: yolo11n.pt (Détection)...")
    model_det = YOLO("yolo11n.pt")
    print("    Modèle détection téléchargé\n")
    
    # Modèle segmentation
    print(" Téléchargement: yolo11n-seg.pt (Segmentation)...")
    model_seg = YOLO("yolo11n-seg.pt")
    print("    Modèle segmentation téléchargé\n")
    
    # Modèle keypoints
    print(" Téléchargement: yolo11n-pose.pt (Keypoints)...")
    model_kpt = YOLO("yolo11n-pose.pt")
    print("    Modèle keypoints téléchargé\n")
    
    print("=" * 60)
    print(" TOUS LES MODÈLES SONT TÉLÉCHARGÉS!")
    print("=" * 60)
    print(f"\n Les modèles sont sauvegardés dans: {Path.home() / '.ultralytics'}")
    
except Exception as e:
    print(f" Erreur lors du téléchargement: {e}")
    print("\n Vérifiez votre connexion internet et réessayez")

