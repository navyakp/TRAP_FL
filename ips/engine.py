import json
import os

BLOCKLIST_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'blocklist.json')

class IPSEngine:
    def __init__(self):
        self.blocklist = set()
        self._load_blocklist()
        
    def _load_blocklist(self):
        if os.path.exists(BLOCKLIST_FILE):
            try:
                with open(BLOCKLIST_FILE, 'r') as f:
                    self.blocklist = set(json.load(f))
            except:
                pass
                
    def _save_blocklist(self):
        with open(BLOCKLIST_FILE, 'w') as f:
            json.dump(list(self.blocklist), f)

    def evaluate_traffic(self, client_ip, prediction_prob, current_risk):
        # Check if already blocked
        if client_ip in self.blocklist:
            return "BLOCK", "Already in blocklist"
            
        # Decision Logic combining prediction and temporal risk
        # Prediction > 0.5 is malicious
        action = "ALLOW"
        reason = "Normal traffic"
        
        if prediction_prob > 0.9 and current_risk > 0.8:
            action = "BLOCK"
            reason = "High confidence malicious activity with high risk"
            self.blocklist.add(client_ip)
            self._save_blocklist()
        elif prediction_prob > 0.7 or current_risk > 0.7:
            action = "ALERT"
            reason = "Suspicious activity detected"
        elif prediction_prob > 0.4 or current_risk > 0.4:
            action = "MONITOR"
            reason = "Elevated risk or borderline prediction"
            
        return action, reason

engine = IPSEngine()
