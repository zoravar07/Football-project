# Quick start (Mac)

## First time

1. Open **Terminal**.
2. Type `cd ` (with a space), drag the **PL project** folder into the Terminal window, press Enter.
   Type `ls` and you should see `run_pipeline.py`, `src`, `dashboard`...
3. Create a virtual environment (a private box for this project's packages):
   ```bash
   python3 -m venv venv
   ```
4. Switch it on. `(venv)` appears at the start of the line:
   ```bash
   source venv/bin/activate
   ```
5. Install the packages (takes a minute or two):
   ```bash
   pip install -r requirements.txt
   ```
6. Download the data and train the models (about a minute):
   ```bash
   python run_pipeline.py
   ```
7. Start the dashboard. It opens in your browser at http://localhost:8501:
   ```bash
   streamlit run dashboard/app.py
   ```
   To stop it, click on Terminal and press **Ctrl + C**.

## Every time after that

```bash
cd   (drag the folder in again)
source venv/bin/activate
python run_pipeline.py        # only when you want the latest results
streamlit run dashboard/app.py
```

## If something goes wrong

| Problem | Fix |
|---|---|
| `command not found: python` | Your venv isn't on. Run `source venv/bin/activate` |
| `No module named ...` | Run `pip install -r requirements.txt` with the venv on |
| `skipped 2005-06: ... CERTIFICATE_VERIFY_FAILED` | Run `pip install --upgrade certifi`, then try again |
| `skipped ...` for every season | No internet, or football-data.co.uk is down. Try again later |
| Dashboard says "No trained models found" | Run `python run_pipeline.py` first |
| `Port 8501 is already in use` | `streamlit run dashboard/app.py --server.port 8502` |
| Red warning: "NotOpenSSLWarning" | Harmless, ignore it |
