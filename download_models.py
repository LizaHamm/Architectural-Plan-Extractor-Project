"""
Script pour télécharger les modèles YOLO
Exécuter avec: python download_models.py
"""

from ultralytics import YOLO
from pathlib import Path

print("=" * 60)
print("🔄 TÉLÉCHARGEMENT DES MODÈLES YOLO")
print("=" * 60)
print("\n⏳ Cela peut prendre quelques minutes lors du premier téléchargement...\n")

try:
    # Modèle détection (le plus utilisé)
    print("📥 Téléchargement: yolo11n.pt (Détection)...")
    model_det = YOLO("yolo11n.pt")
    print("   ✅ Modèle détection téléchargé\n")
    
    # Modèle segmentation
    print("📥 Téléchargement: yolo11n-seg.pt (Segmentation)...")
    model_seg = YOLO("yolo11n-seg.pt")
    print("   ✅ Modèle segmentation téléchargé\n")
    
    # Modèle keypoints
    print("📥 Téléchargement: yolo11n-pose.pt (Keypoints)...")
    model_kpt = YOLO("yolo11n-pose.pt")
    print("   ✅ Modèle keypoints téléchargé\n")
    
    print("=" * 60)
    print("✅ TOUS LES MODÈLES SONT TÉLÉCHARGÉS!")
    print("=" * 60)
    print(f"\n💡 Les modèles sont sauvegardés dans: {Path.home() / '.ultralytics'}")
    
except Exception as e:
    print(f"❌ Erreur lors du téléchargement: {e}")
    print("\n💡 Vérifiez votre connexion internet et réessayez")

