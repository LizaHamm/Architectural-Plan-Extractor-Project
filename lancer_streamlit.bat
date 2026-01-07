@echo off
echo ========================================
echo  Lancement de l'interface Streamlit
echo ========================================
echo.

REM Activer l'environnement virtuel
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo Environnement virtuel active
) else (
    echo Attention: Environnement virtuel non trouve
    echo Assurez-vous d'avoir cree et active votre venv
)

echo.
echo Demarrage de Streamlit...
echo.
echo L'application s'ouvrira dans votre navigateur
echo Appuyez sur Ctrl+C pour arreter
echo.

REM Utiliser python -m streamlit pour plus de fiabilite
python -m streamlit run app_streamlit.py

pause

