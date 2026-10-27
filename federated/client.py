import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np

class FLClient:
    def __init__(self, client_id, model, data_x, data_y, lr=0.01, epochs=1, batch_size=32):
        self.client_id = client_id
        self.model = model
        self.data_x = torch.FloatTensor(data_x)
        self.data_y = torch.FloatTensor(data_y).view(-1, 1)
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.criterion = nn.BCELoss()
        
        # TRAP-FL risk metrics
        self.current_risk = 0.0
        self.temporal_risk = 0.0
        
    def set_weights(self, global_weights):
        self.model.load_state_dict(copy.deepcopy(global_weights))
        
    def train(self):
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        dataset = torch.utils.data.TensorDataset(self.data_x, self.data_y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        all_preds = []
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                # Collect predictions to calculate current risk
                if epoch == self.epochs - 1:
                    all_preds.extend(outputs.detach().numpy().flatten())
                    
        # Update current risk: average probability of malicious traffic
        # If the client processes highly malicious traffic, risk goes up
        self.current_risk = np.mean(all_preds) if len(all_preds) > 0 else 0.0
        
        return self.model.state_dict(), len(self.data_x), self.current_risk
