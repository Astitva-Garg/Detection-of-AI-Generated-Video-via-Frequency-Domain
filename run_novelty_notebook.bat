@echo off
rem Opens novelty.ipynb, the novelty-assessment notebook. First run creates .venv-app and installs requirements-app.txt.
cd /d "%~dp0"
if not exist ".venv-app\Scripts\python.exe" (
  echo Creating the environment, first run only. This takes a few minutes...
  py -3.11 -m venv .venv-app
  if errorlevel 1 (
    echo Python 3.11 was not found. Install it from python.org, then run this again.
    exit /b 1
  )
  ".venv-app\Scripts\python.exe" -m pip install --upgrade pip
  ".venv-app\Scripts\python.exe" -m pip install -r requirements-app.txt
  if errorlevel 1 (
    echo Install failed. See the messages above.
    exit /b 1
  )
)
".venv-app\Scripts\python.exe" -m ipykernel install --sys-prefix --name ie643 --display-name "IE643 demo" >nul
echo Opening novelty.ipynb. Run the cells one at a time, top to bottom.
".venv-app\Scripts\python.exe" -m notebook novelty.ipynb
