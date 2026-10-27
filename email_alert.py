import smtplib
import logging
import threading
import queue
import time
import uuid
import email.utils
import os
from datetime import datetime
from email.header import Header
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

logger = logging.getLogger(__name__)

# ── SMTP Configuration ────────────────────────────────────────────────────────
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT_SSL = 465
SMTP_PORT_TLS = 587
SENDER_EMAIL = os.environ.get("TRAPFL_SMTP_EMAIL", "")
SENDER_PASSWORD = os.environ.get("TRAPFL_SMTP_PASSWORD", "")
SENDER_NAME = "TRAP-FL Security System"

# Dedicated background queue to prevent overlapping SMTP connection collisions
_alert_queue = queue.Queue()
_worker_thread = None
_worker_lock = threading.Lock()


def build_alert_html(client_ip, probability, ips_action, reason, features_dict, username, timestamp_str):
    prob_pct = f"{probability * 100:.2f}%"
    
    feature_rows = ""
    for k, v in features_dict.items():
        feature_rows += f"""
        <tr>
            <td style="padding: 7px 12px; border: 1px solid #e2e8f0; font-family: monospace; font-size: 13px; color: #334155; font-weight: 600;">{k}</td>
            <td style="padding: 7px 12px; border: 1px solid #e2e8f0; font-family: monospace; font-size: 13px; color: #0f172a; text-align: right;">{v}</td>
        </tr>
        """

    action_colors = {
        "BLOCK": "#dc2626",
        "ALERT": "#ea580c",
        "MONITOR": "#0284c7",
        "ALLOW": "#16a34a"
    }
    action_color = action_colors.get(ips_action.upper(), "#dc2626")

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>TRAP-FL Security Alert</title>
</head>
<body style="margin:0; padding:0; background-color:#f1f5f9; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background-color:#f1f5f9; padding: 25px 0;">
    <tr>
      <td align="center">
        <table width="600" cellpadding="0" cellspacing="0" style="background:#ffffff; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.08); border: 1px solid #e2e8f0;">
          
          <!-- Alert Header Banner -->
          <tr>
            <td style="background: linear-gradient(135deg, #991b1b, #ef4444); padding: 24px 30px; text-align: center;">
              <h1 style="margin: 0; color: #ffffff; font-size: 22px; font-weight: 800; letter-spacing: 0.5px;">
                SECURITY ALERT: MALICIOUS TRAFFIC DETECTED
              </h1>
              <p style="margin: 8px 0 0; color: #fee2e2; font-size: 14px;">
                TRAP-FL Intrusion Detection &amp; Prevention System
              </p>
            </td>
          </tr>

          <!-- Summary Box -->
          <tr>
            <td style="padding: 24px 30px;">
              <p style="margin: 0 0 16px; font-size: 15px; color: #334155; line-height: 1.5;">
                Anomalous and potentially dangerous network traffic has been classified as <strong>MALICIOUS</strong> by the TRAP-FL Manual Prediction Engine.
              </p>
              
              <table width="100%" cellpadding="0" cellspacing="0" style="background-color: #fef2f2; border: 1.5px solid #fecaca; border-radius: 8px; margin-bottom: 20px;">
                <tr>
                  <td style="padding: 16px 20px;">
                    <table width="100%" cellpadding="4" cellspacing="0">
                      <tr>
                        <td width="35%" style="font-size: 13px; color: #64748b; font-weight: 600;">Detection Status:</td>
                        <td style="font-size: 14px; font-weight: 700; color: #b91c1c;">MALICIOUS THREAT DETECTED</td>
                      </tr>
                      <tr>
                        <td style="font-size: 13px; color: #64748b; font-weight: 600;">Threat Probability:</td>
                        <td style="font-size: 14px; font-weight: 700; color: #b91c1c;">{prob_pct}</td>
                      </tr>
                      <tr>
                        <td style="font-size: 13px; color: #64748b; font-weight: 600;">Target Client IP:</td>
                        <td style="font-size: 14px; font-family: monospace; font-weight: 700; color: #0f172a;">{client_ip}</td>
                      </tr>
                      <tr>
                        <td style="font-size: 13px; color: #64748b; font-weight: 600;">IPS Action:</td>
                        <td>
                          <span style="display:inline-block; background-color:{action_color}; color:#ffffff; font-weight:bold; font-size:12px; padding:3px 10px; border-radius:4px;">
                            {ips_action}
                          </span>
                        </td>
                      </tr>
                      <tr>
                        <td style="font-size: 13px; color: #64748b; font-weight: 600;">Prevention Reason:</td>
                        <td style="font-size: 13px; color: #334155;">{reason}</td>
                      </tr>
                      <tr>
                        <td style="font-size: 13px; color: #64748b; font-weight: 600;">Triggered By User:</td>
                        <td style="font-size: 13px; color: #334155;">{username}</td>
                      </tr>
                      <tr>
                        <td style="font-size: 13px; color: #64748b; font-weight: 600;">Timestamp:</td>
                        <td style="font-size: 13px; color: #64748b;">{timestamp_str}</td>
                      </tr>
                    </table>
                  </td>
                </tr>
              </table>

              <!-- Network Parameters Table -->
              <h3 style="margin: 0 0 10px; font-size: 14px; color: #0f172a; text-transform: uppercase; letter-spacing: 0.5px;">
                Submitted Network Traffic Parameters
              </h3>
              <table width="100%" cellpadding="0" cellspacing="0" style="border-collapse: collapse; border: 1px solid #e2e8f0; border-radius: 6px; overflow: hidden; margin-bottom: 20px;">
                <thead>
                  <tr style="background-color: #f8fafc;">
                    <th style="padding: 8px 12px; border: 1px solid #e2e8f0; font-size: 12px; text-align: left; color: #64748b; text-transform: uppercase;">Feature</th>
                    <th style="padding: 8px 12px; border: 1px solid #e2e8f0; font-size: 12px; text-align: right; color: #64748b; text-transform: uppercase;">Observed Value</th>
                  </tr>
                </thead>
                <tbody>
                  {feature_rows}
                </tbody>
              </table>

              <div style="background-color: #f8fafc; border-left: 4px solid #0284c7; padding: 12px 16px; border-radius: 4px;">
                <p style="margin: 0; font-size: 12px; color: #475569; line-height: 1.5;">
                  <strong>Action Required:</strong> The IPS Engine has taken the enforcement action above. Please inspect network activity associated with IP <code>{client_ip}</code> in your TRAP-FL dashboard.
                </p>
              </div>

            </td>
          </tr>

          <!-- Footer -->
          <tr>
            <td style="background-color: #f8fafc; padding: 16px 30px; text-align: center; border-top: 1px solid #e2e8f0;">
              <p style="margin: 0; font-size: 12px; color: #94a3b8;">
                Automated notification sent by TRAP-FL Security Monitoring System. Do not reply directly.
              </p>
            </td>
          </tr>

        </table>
      </td>
    </tr>
  </table>
</body>
</html>"""
    return html


def build_alert_plain(client_ip, probability, ips_action, reason, features_dict, username, timestamp_str):
    prob_pct = f"{probability * 100:.2f}%"
    feats = "\n".join([f"  - {k}: {v}" for k, v in features_dict.items()])
    return f"""====================================================
TRAP-FL SECURITY ALERT: MALICIOUS TRAFFIC DETECTED
====================================================

Detection Status   : MALICIOUS THREAT DETECTED
Threat Probability : {prob_pct}
Target Client IP   : {client_ip}
IPS Action         : {ips_action}
Action Reason      : {reason}
Triggered By       : {username}
Timestamp          : {timestamp_str}

Traffic Parameters:
{feats}

Action Required:
Inspect network activity associated with IP {client_ip} in your TRAP-FL dashboard.

This is an automated alert sent by TRAP-FL Security Monitoring System.
"""


def _get_smtp_connection():
    """
    Connects to Gmail SMTP using SSL (port 465) first, falling back to STARTTLS (port 587).
    """
    try:
        server = smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT_SSL, timeout=25)
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        return server
    except Exception as ssl_err:
        logger.warning(f"Port 465 SSL connection failed ({ssl_err}), trying port 587 TLS...")

    server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT_TLS, timeout=25)
    server.ehlo()
    server.starttls()
    server.ehlo()
    server.login(SENDER_EMAIL, SENDER_PASSWORD)
    return server


def _deliver_email_with_retry(client_ip, probability, ips_action, reason, features_dict, valid_recipients, username, max_retries=2):
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    time_display = datetime.now().strftime("%I:%M:%S %p")
    user_display = username or "Operator"
    prob_pct = f"{probability * 100:.2f}%"

    html_content = build_alert_html(client_ip, probability, ips_action, reason, features_dict, user_display, timestamp_str)
    plain_content = build_alert_plain(client_ip, probability, ips_action, reason, features_dict, user_display, timestamp_str)

    msg = MIMEMultipart("alternative")
    msg["From"] = f"{SENDER_NAME} <{SENDER_EMAIL}>"
    msg["To"] = ", ".join(valid_recipients)
    
    # Use Header(..., 'utf-8') to avoid ASCII/charmap encoding issues in email headers
    subject_text = f"[TRAP-FL ALERT] Malicious Traffic [{client_ip}] ({prob_pct}) - {time_display}"
    msg["Subject"] = Header(subject_text, "utf-8")
    msg["Message-ID"] = f"<trapfl.{int(time.time())}.{uuid.uuid4().hex[:8]}@smtp.gmail.com>"
    msg["Date"] = email.utils.formatdate(localtime=True)
    msg["X-Priority"] = "1"

    msg.attach(MIMEText(plain_content, "plain", "utf-8"))
    msg.attach(MIMEText(html_content, "html", "utf-8"))

    raw_msg = msg.as_string()

    for attempt in range(1, max_retries + 1):
        try:
            print(f"[TRAP-FL ALERT] (Attempt {attempt}/{max_retries}) Connecting to Gmail SMTP for: {', '.join(valid_recipients)}...", flush=True)
            server = _get_smtp_connection()
            with server:
                server.sendmail(SENDER_EMAIL, valid_recipients, raw_msg)
            print(f"[TRAP-FL ALERT] [OK] Successfully delivered alert to: {', '.join(valid_recipients)}", flush=True)
            return True, valid_recipients, None
        except Exception as err:
            print(f"[TRAP-FL ALERT] [WARNING] Delivery attempt {attempt} failed: {err}", flush=True)
            if attempt < max_retries:
                time.sleep(2)
            else:
                print(f"[TRAP-FL ALERT] [ERROR] All {max_retries} delivery attempts failed: {err}", flush=True)
                return False, [], str(err)


def _worker_loop():
    """Background queue worker to process email dispatching serially and prevent socket collisions."""
    while True:
        try:
            task = _alert_queue.get()
            if task is None:
                break
            client_ip, probability, ips_action, reason, features_dict, valid_recipients, username = task
            _deliver_email_with_retry(client_ip, probability, ips_action, reason, features_dict, valid_recipients, username)
        except Exception as e:
            print(f"[TRAP-FL ALERT] [ERROR] Worker loop exception: {e}", flush=True)
        finally:
            _alert_queue.task_done()


def _ensure_worker_running():
    global _worker_thread
    with _worker_lock:
        if _worker_thread is None or not _worker_thread.is_alive():
            _worker_thread = threading.Thread(target=_worker_loop, daemon=True)
            _worker_thread.start()


def send_malicious_alert(client_ip, probability, ips_action, reason, features_dict, recipients, username=None, async_send=True):
    """
    Sends an automated email alert when malicious traffic is detected in manual prediction.
    Uses a robust queued worker with retries and single-call multi-recipient delivery.
    """
    if isinstance(recipients, str):
        recipients = [recipients]

    valid_recipients = []
    for r in recipients:
        r_clean = (r or "").strip()
        if r_clean and "@" in r_clean and r_clean not in valid_recipients:
            valid_recipients.append(r_clean)

    if not valid_recipients:
        return {
            "sent": False,
            "recipients": [],
            "failed": [],
            "message": "No valid recipient email address provided."
        }
    if not SENDER_EMAIL or not SENDER_PASSWORD:
        raise RuntimeError(
            "Email alerts require TRAPFL_SMTP_EMAIL and TRAPFL_SMTP_PASSWORD."
        )

    if async_send:
        _ensure_worker_running()
        _alert_queue.put((client_ip, probability, ips_action, reason, features_dict, valid_recipients, username))
        print(f"[TRAP-FL ALERT] Queued security alert for {', '.join(valid_recipients)} (Queue size: {_alert_queue.qsize()})", flush=True)
        return {
            "sent": True,
            "recipients": valid_recipients,
            "failed": [],
            "message": f"Security alert dispatched to {', '.join(valid_recipients)}"
        }
    else:
        success, delivered, error_str = _deliver_email_with_retry(
            client_ip, probability, ips_action, reason, features_dict, valid_recipients, username
        )
        return {
            "sent": success,
            "recipients": delivered,
            "failed": [] if success else valid_recipients,
            "message": f"Alert sent to {', '.join(delivered)}" if success else f"Failed to deliver alert: {error_str}"
        }
