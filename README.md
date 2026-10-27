# TRAP-FL

TRAP-FL is a federated-learning intrusion detection and prevention project.

## Setup

Use Python 3.10 or newer, create and activate a virtual environment, and install
the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Set the Flask signing key before starting the application:

```powershell
$env:FLASK_SECRET_KEY = "<generate-a-long-random-value>"
```

To send email alerts, also set the SMTP sender credentials and a comma-separated
list of alert recipients:

```powershell
$env:TRAPFL_SMTP_EMAIL = "<sender-address>"
$env:TRAPFL_SMTP_PASSWORD = "<smtp-app-password>"
$env:TRAPFL_ALERT_RECIPIENTS = "<recipient-addresses>"
```

Run the application with `python app.py`. Do not commit credentials, `.env`
files, or the local SQLite database.
