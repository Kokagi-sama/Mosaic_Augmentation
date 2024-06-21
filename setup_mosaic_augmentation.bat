@echo off
echo Setting up and running the mosaic augmentation application...

:: Check if the .venv directory already exists
if not exist .venv (
    # Create virtual environment
    python -m venv .venv
) else (
    echo Virtual environment already exists. Skipping creation.
)

:: Activate the virtual environment
call .venv\Scripts\activate

:: Install the dependencies
pip install -r requirements.txt

:: Run the Mosaic Augmentation Script
python mosaic_augmentation.py

pause
