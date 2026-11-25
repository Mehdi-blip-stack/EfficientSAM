# EfficientSAM Project Setup

## Requirements
- Python 3.14, 64-bit
- Git
- VS Code
- Our shared COCO subset
## Create the Python environment
Open PowerShell in the TinySam project folder and run:
```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-eda.txt
```
Each teammate creates their own environment. Do not copy or upload `.venv`.

## Set the dataset location
Copy or extract our shared dataset to a folder on your computer.
It should contain:
```text
COCO2017_EfficientSAM_500/
    annotations/
    instances_train2017.json
    instances_val2017.json
    subset_instances_train2017.json
    subset_instances_val2017.json
  images/
    train2017/
    val2017/
  subset_selection_1000.json
```
The folder name says 500 for historical reasons; the subset contains
800 training images and 200 evaluation images.
In PowerShell, set the location. Replace this example with your own path:
```powershell
$env:COCO_DATA_ROOT = "D:\DataSet\COCO2017_EfficientSAM_500"
```
Set this again whenever you open a new terminal.
Check that the training annotations exist:
```powershell
Test-Path "$env:COCO_DATA_ROOT\annotations\subset_instances_train2017.json"
```
The result should be `True`.
## Run the EDA
```powershell
.\.venv\Scripts\python.exe scripts\eda_coco_subset.py
```
Close each chart window to let the script continue.
## Generate the reports
First set `COCO_DATA_ROOT` as described above.
Generate the report charts and summary:
```powershell
.\.venv\Scripts\python.exe scripts\export_coco_eda_progress.py chart
```
Generate the four-page PDF:
```powershell
.\.venv\Scripts\python.exe scripts\export_coco_eda_progress.py pdf
```
Generate the expanded editable Word report:
```powershell
.\.venv\Scripts\python.exe scripts\build_editable_eda.py
```
The generated files are saved under `output/`.
To create a PDF from the Word report, open it in Microsoft Word
and choose File → Save As → PDF.