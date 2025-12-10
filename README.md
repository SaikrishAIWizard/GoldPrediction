**Project Overview**

- **Name:** GoldPrediction
- **Purpose:** A Django app for forecasting gold prices and running model backtests.

**Quick Start**

- **Create & activate virtualenv (PowerShell):**

```powershell
# create virtual environment (if needed)
python -m venv venv
# activate
.\venv\Scripts\Activate.ps1
# install dependencies
pip install -r requirements.txt
```

- **Run the development server:**

```powershell
cd myproject
python manage.py migrate
python manage.py runserver
```

**Project Layout (key items)**

- **Django project:** `myproject/`
- **App:** `myproject/firstapp/`
- **Data & models:** `myproject/firstapp/Models/` (trained models in `Models/gold/`)
- **Main manage command:** `myproject/manage.py`

**Where to find trained models & metrics**

- Models and metrics are stored in `myproject/firstapp/Models/gold/`.

**Notes & Next Steps**

- To update or retrain models, check `myproject/firstapp/Models/` and the training views in `myproject/firstapp/views.py`.
- Consider adding a `requirements.txt` lock (e.g., `pip freeze > requirements.txt`) and a short CONTRIBUTING note.

**License**

- This repository doesn't include a license file. Add one if you plan to publish or share.

python 3.10.0
