import os
import time
import json
import pickle
import pandas as pd
import numpy as np
import torch
from flask import Flask, render_template, request, jsonify, Response, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash

from models.base_model import get_model
from ips.engine import IPSEngine
from database import (init_db, get_user_by_id, get_user_by_username,
                      create_user, save_prediction, get_user_history,
                      delete_history_item, clear_user_history)
from email_alert import send_malicious_alert


CONCERNED_AUTHORITY_EMAILS = [
    email.strip()
    for email in os.environ.get('TRAPFL_ALERT_RECIPIENTS', '').split(',')
    if email.strip()
]
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY')
if not app.secret_key:
    raise RuntimeError("FLASK_SECRET_KEY must be set to start the application.")


login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'warning'


class User(UserMixin):
    def __init__(self, data):
        self.id = data['id']
        self.username = data['username']
        self.email = data['email']


@login_manager.user_loader
def load_user(user_id):
    data = get_user_by_id(int(user_id))
    return User(data) if data else None


# Explicit context processor — guarantees current_user is always in every template
@app.context_processor
def inject_globals():
    return dict(current_user=current_user, request=request)



init_db()


BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, 'saved_models')
DATA_DIR  = os.path.join(BASE_DIR, 'data')


try:
    with open(os.path.join(MODEL_DIR, 'scaler_full.pkl'), 'rb') as f:
        scaler_full = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'scaler_8.pkl'), 'rb') as f:
        scaler_8 = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'features_full.pkl'), 'rb') as f:
        features_full = pickle.load(f)
except Exception as e:
    print("Warning: Scalers/features not found. Run experiments/runner.py first.")
    scaler_full, scaler_8, features_full = None, None, []


try:
    model_full = get_model(len(features_full))
    model_full.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'trap_fl_model.pth'), weights_only=True))
    model_full.eval()

    model_manual = get_model(8)
    model_manual.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'manual_model.pth'), weights_only=True))
    model_manual.eval()
except Exception as e:
    print("Warning: Could not load models:", e)
    model_full, model_manual = None, None

ips_engine = IPSEngine()

sim_state = {"is_running": False, "row_index": 0, "test_data": None}


def load_test_data():
    if sim_state["test_data"] is None:
        try:
            df = pd.read_csv(os.path.join(DATA_DIR, 'UNSW_NB15_testing-set.csv'))
            if 'id' in df.columns:         df.drop('id', axis=1, inplace=True)
            if 'attack_cat' in df.columns: df.drop('attack_cat', axis=1, inplace=True)
            with open(os.path.join(MODEL_DIR, 'label_encoders.pkl'), 'rb') as f:
                le_dict = pickle.load(f)
            for col, le in le_dict.items():
                if col in df.columns:
                    df[col] = df[col].astype(str).apply(lambda x: x if x in le.classes_ else le.classes_[0])
                    df[col] = le.transform(df[col])
            sim_state["test_data"] = df
        except Exception as e:
            print("Error loading test data:", e)




@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        user_data = get_user_by_username(username)
        if user_data and check_password_hash(user_data['password_hash'], password):
            user = User(user_data)
            login_user(user, remember=request.form.get('remember') == 'on')
            flash(f'Welcome back, {user.username}!', 'success')
            return redirect(request.args.get('next') or url_for('dashboard'))
        flash('Invalid username or password.', 'danger')
    return render_template('login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email    = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm  = request.form.get('confirm_password', '')
        if not all([username, email, password]):
            flash('All fields are required.', 'danger')
        elif len(username) < 3:
            flash('Username must be at least 3 characters.', 'danger')
        elif password != confirm:
            flash('Passwords do not match.', 'danger')
        elif len(password) < 6:
            flash('Password must be at least 6 characters.', 'danger')
        elif create_user(username, email, generate_password_hash(password)):
            flash('Account created! Please log in.', 'success')
            return redirect(url_for('login'))
        else:
            flash('Username or email already exists.', 'danger')
    return render_template('signup.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))




@app.route('/dashboard')
@login_required
def dashboard():
    results = []
    path = os.path.join(DATA_DIR, 'experiment_results.csv')
    if os.path.exists(path):
        results = pd.read_csv(path).to_dict('records')
    return render_template('dashboard.html', results=results)


@app.route('/manual')
@login_required
def manual_prediction():
    features = ['sttl', 'ct_state_ttl', 'sload', 'rate', 'sbytes', 'dbytes', 'dload', 'smean']
    return render_template('manual.html', features=features)


@app.route('/api/predict_manual', methods=['POST'])
@login_required
def api_predict_manual():
    data = request.json or {}
    FEAT = ['sttl', 'ct_state_ttl', 'sload', 'rate', 'sbytes', 'dbytes', 'dload', 'smean']
    vals = [float(data.get(f, 0.0)) for f in FEAT]
    scaled = scaler_8.transform([vals])
    with torch.no_grad():
        prob = model_manual(torch.FloatTensor(scaled)).item()

    client_ip = data.get('client_ip', '192.168.1.100')
    action, reason = ips_engine.evaluate_traffic(
        client_ip, prob, current_risk=prob)
    ids_cls = 'Malicious' if prob > 0.5 else 'Normal'

    save_prediction(
        user_id=current_user.id,
        prediction_type='manual',
        input_data=json.dumps({f: data.get(f, 0) for f in FEAT}),
        probability=round(prob, 6),
        ids_classification=ids_cls,
        ips_action=action,
        reason=reason
    )

    email_status = None
    if prob > 0.5:
        feats_dict = {f: data.get(f, 0) for f in FEAT}
        username = getattr(current_user, 'username', 'Analyst')
        try:
            email_status = send_malicious_alert(
                client_ip=client_ip,
                probability=prob,
                ips_action=action,
                reason=reason,
                features_dict=feats_dict,
                recipients=CONCERNED_AUTHORITY_EMAILS,
                username=username
            )
        except Exception as em_err:
            email_status = {
                "sent": False,
                "recipients": [],
                "failed": CONCERNED_AUTHORITY_EMAILS,
                "message": f"Failed to send alert email: {str(em_err)}"
            }

    return jsonify({
        "prediction_prob":   prob,
        "is_malicious":      prob > 0.5,
        "ids_classification": ids_cls,
        "action": action,
        "reason": reason,
        "email_status": email_status
    })


@app.route('/simulation')
@login_required
def simulation():
    return render_template('simulation.html')


@app.route('/api/sim/control', methods=['POST'])
@login_required
def sim_control():
    cmd = request.json.get('command')
    if cmd == 'START':
        load_test_data()
        sim_state["is_running"] = True
    elif cmd == 'PAUSE':
        sim_state["is_running"] = False
    elif cmd == 'STOP':
        sim_state["is_running"] = False
        sim_state["row_index"]  = 0
    return jsonify({"status": sim_state["is_running"]})


@app.route('/api/sim/stream')
@login_required
def sim_stream():
    def generate():
        while True:
            if sim_state["is_running"] and sim_state["test_data"] is not None:
                if sim_state["row_index"] < len(sim_state["test_data"]):
                    row = sim_state["test_data"].iloc[sim_state["row_index"]]
                    features    = row.drop('label').values.reshape(1, -1)
                    true_label  = int(row['label'])
                    scaled      = scaler_full.transform(features)
                    with torch.no_grad():
                        prob = model_full(torch.FloatTensor(scaled)).item()
                    action, reason = ips_engine.evaluate_traffic("10.0.0.5", prob, current_risk=prob)
                    payload = json.dumps({
                        'index':      sim_state['row_index'],
                        'prob':       prob,
                        'true_label': true_label,
                        'action':     action,
                        'reason':     reason
                    })
                    yield f"data: {payload}\n\n"
                    sim_state["row_index"] += 1
                    time.sleep(1)
                else:
                    sim_state["is_running"] = False
            else:
                time.sleep(0.5)
    return Response(generate(), mimetype='text/event-stream')


@app.route('/history')
@login_required
def history():
    records = get_user_history(current_user.id)
    return render_template('history.html', records=records)


@app.route('/api/history/delete/<int:item_id>', methods=['DELETE'])
@login_required
def delete_history(item_id):
    delete_history_item(item_id, current_user.id)
    return jsonify({"status": "deleted"})


@app.route('/api/history/clear', methods=['DELETE'])
@login_required
def clear_history():
    clear_user_history(current_user.id)
    return jsonify({"status": "cleared"})


@app.route('/api/ips/unblock', methods=['POST'])
@login_required
def unblock_ip():
    ip = request.json.get('ip')
    if ip and ip in ips_engine.blocklist:
        ips_engine.blocklist.remove(ip)
        ips_engine._save_blocklist()
    return jsonify({"status": "unblocked", "ip": ip})




if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=False)
