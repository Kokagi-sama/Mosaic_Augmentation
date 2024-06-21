# Project: Mosaic_Augmentation



## Overview
This project showcases simple mosaic augmentation.

## Prerequisites
Python (preferably Python 3.x)
IDE or Text Editor (e.g., Visual Studio Code, PyCharm)

## Clone the Repository:
```
git clone https://github.com/Kokagi-sama/Mosaic_Augmentation.git
```

## Easy Setup
Run the batch file `setup_mosaic_augmentation.bat`.

## Manual Setup
### Python
1. Go to the directory where you have cloned the repository.

2. Setup Python Virtual Environment at root directory:
```
python -m venv .venv
.venv\Scripts\activate 
```

3. Install Python Dependencies:
```
pip install -r requirements.txt
```

4. Run the Mosaic Augmentation Script:
```
python mosaic_augmentation.py
```

## Usage:
1. Input the output size desired (e.g. 416 will result in final output of 416x416 dimensions.)

2. Input the partition ratio of horizontal plane/x plane (0.0 < x < 1.0) (e.g. x_ratio = 0.3 will result in 1st and 3rd quadrant occupying 0.3 of the final image and 2nd and 4th quadrant occupying 0.7 of the rest of the final image.)

3. Input the partition ratio of vertical plane/y plane (0.0 < y < 1.0) (e.g. y_ratio = 0.4 will result in 1st and 2nd quadrant occupying 0.4 of the final image and 3rd and 4th quadrant occupying 0.6 of the rest of the final image.)

4. Enjoy the final output.
