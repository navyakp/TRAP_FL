import torch
import copy
import numpy as np

class BaseServer:
    def __init__(self, global_model):
        self.global_model = global_model
        
    def aggregate(self, client_updates):
        total_samples = sum([n for _, n, _ in client_updates])
        
        global_dict = self.global_model.state_dict()
        for k in global_dict.keys():
            global_dict[k] = torch.zeros_like(global_dict[k], dtype=torch.float)
            
        for client_weights, n_samples, _ in client_updates:
            weight = n_samples / total_samples
            for k in global_dict.keys():
                global_dict[k] += client_weights[k] * weight
                
        self.global_model.load_state_dict(global_dict)
        return self.global_model.state_dict()
