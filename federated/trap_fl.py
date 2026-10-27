import torch
import copy
import numpy as np

class TRAP_FL_Server:
    def __init__(self, global_model, alpha=0.5, c_base=1.0, beta=0.5, sigma_base=0.01, gamma=1.0):
        self.global_model = global_model
        
        # TRAP-FL hyperparams
        self.alpha = alpha       # Temporal decay
        self.c_base = c_base     # Base clipping threshold
        self.beta = beta         # Risk impact on clipping
        self.sigma_base = sigma_base # Base noise
        self.gamma = gamma       # Risk impact on noise
        
        # Historical risks
        self.client_temporal_risks = {}
        
    def _apply_dp(self, client_id, model_update, temporal_risk):
        # 1. Adapt clipping threshold
        C_it = self.c_base * (1 - self.beta * temporal_risk)
        C_it = max(C_it, 0.01) # Avoid zero or negative clipping
        
        # 2. Adapt noise scale
        sigma_it = self.sigma_base * (1 + self.gamma * temporal_risk)
        
        # Flatten update to calculate L2 norm
        flat_update = torch.cat([v.flatten() for v in model_update.values()])
        l2_norm = torch.norm(flat_update, p=2)
        
        # 3. Clip
        clip_factor = min(1.0, C_it / (l2_norm.item() + 1e-10))
        clipped_update = {}
        
        for k, v in model_update.items():
            clipped_v = v * clip_factor
            # 4. Add Gaussian Noise
            noise = torch.randn_like(clipped_v) * sigma_it
            clipped_update[k] = clipped_v + noise
            
        return clipped_update

    def aggregate(self, client_updates):
        total_samples = sum([n for _, n, _ in client_updates])
        
        global_dict = self.global_model.state_dict()
        new_global_dict = {k: torch.zeros_like(v, dtype=torch.float) for k, v in global_dict.items()}
        
        for client_id, (client_weights, n_samples, current_risk) in enumerate(client_updates):
            # 1. Update temporal risk (EMA)
            prev_risk = self.client_temporal_risks.get(client_id, 0.0)
            temporal_risk = self.alpha * current_risk + (1 - self.alpha) * prev_risk
            self.client_temporal_risks[client_id] = temporal_risk
            
            # Calculate delta weight (model update)
            delta_w = {k: client_weights[k] - global_dict[k] for k in global_dict.keys()}
            
            # 2. Apply Risk-Adaptive DP
            dp_update = self._apply_dp(client_id, delta_w, temporal_risk)
            
            # Secure Aggregation simulation (masking)
            # In a real SecAgg, clients add pairwise masks that cancel out on the server.
            # Here we simulate the *result* of SecAgg by just securely summing the DP updates.
            # (No explicit pairwise mask computation for simplicity in prototype, just sum)
            
            weight = n_samples / total_samples
            for k in global_dict.keys():
                new_global_dict[k] += (global_dict[k] + dp_update[k]) * weight
                
        self.global_model.load_state_dict(new_global_dict)
        return self.global_model.state_dict()
