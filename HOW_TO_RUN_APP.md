# How to Run the Streamlit App (Windows)

This project runs the app on port **8502** by default.

## Quick Start (recommended)

From the project root folder:

```powershell
.\run_app.bat
```

Then open:

- http://localhost:8502

## PowerShell Alternative

If you prefer PowerShell script launch:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_app.ps1 -Headless
```

Then open:

- http://localhost:8502

## If App Fails to Start

### 1) Script execution policy error (`PSSecurityException`)
Use one of these:

- Use batch launcher (no policy change needed):

```powershell
.\run_app.bat
```

- Or run PS script with policy bypass:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_app.ps1 -Headless
```

### 2) Missing virtual environment / dependencies
Create and install requirements:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3) Port already in use (8502)
Start on another port:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_app.ps1 -Port 8503 -Headless
```

Open:

- http://localhost:8503

## Stop App on Default Port

```powershell
.\run_app.bat stop
```
