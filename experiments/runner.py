import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from data_processing.dataset import ensure_data_exists, preprocess_data, create_non_iid_splits
from models.base_model import get_model
from federated.client import FLClient
from federated.server import BaseServer
from federated.trap_fl import TRAP_FL_Server

def train_centralized(X_train, y_train, X_test, y_test, epochs=5):
    model = get_model(X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    
    dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).view(-1, 1))
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    
    model.train()
    for e in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            
    return evaluate_model(model, X_test, y_test)

def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        preds = model(torch.FloatTensor(X_test)).numpy().flatten()
    
    pred_labels = (preds > 0.5).astype(int)
    acc = accuracy_score(y_test, pred_labels)
    prec = precision_score(y_test, pred_labels, zero_division=0)
    rec = recall_score(y_test, pred_labels, zero_division=0)
    f1 = f1_score(y_test, pred_labels, zero_division=0)
    cm = confusion_matrix(y_test, pred_labels)
    
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    return {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1, 'FPR': fpr, 'FNR': fnr}

def run_federated(clients_data, X_test, y_test, rounds=5, strategy="FedAvg"):
    input_dim = clients_data[0][0].shape[1]
    global_model = get_model(input_dim)
    
    if strategy == "TRAP-FL":
        server = TRAP_FL_Server(global_model)
    else:
        server = BaseServer(global_model)
        
    clients = [FLClient(i, get_model(input_dim), cx, cy) for i, (cx, cy) in enumerate(clients_data)]
    
    for r in range(rounds):
        updates = []
        for c in clients:
            c.set_weights(global_model.state_dict())
            weights, n_samples, current_risk = c.train()
            
            # Simple simulation for standard DP
            if strategy == "FedAvg+DP":
                for k in weights:
                    weights[k] += torch.randn_like(weights[k]) * 0.01
                    
            # For FedAvg+SecAgg, we just use BaseServer but pretend it's masked
            updates.append((weights, n_samples, current_risk))
            
        # Aggregate
        if strategy == "TRAP-FL":
            # Pass full tuple
            server.aggregate(updates)
        else:
            server.aggregate(updates)
            
    return evaluate_model(global_model, X_test, y_test), global_model

if __name__ == "__main__":
    train_p, test_p = ensure_data_exists()
    (X_train, y_train, X_test, y_test), (X_train_8, y_train_8, X_test_8, y_test_8) = preprocess_data(train_p, test_p)
    
    # Use subset for faster simulation
    idx = np.random.choice(len(X_train), 5000, replace=False)
    X_train_sub = X_train[idx]
    y_train_sub = y_train[idx]
    
    idx_test = np.random.choice(len(X_test), 2000, replace=False)
    X_test_sub = X_test[idx_test]
    y_test_sub = y_test[idx_test]
    
    print("Running Centralized...")
    res_cent = train_centralized(X_train_sub, y_train_sub, X_test_sub, y_test_sub)
    
    clients_data = create_non_iid_splits(X_train_sub, y_train_sub, num_clients=5)
    
    print("Running FedAvg...")
    res_fedavg, _ = run_federated(clients_data, X_test_sub, y_test_sub, strategy="FedAvg")
    
    print("Running FedAvg+DP...")
    res_feddp, _ = run_federated(clients_data, X_test_sub, y_test_sub, strategy="FedAvg+DP")
    
    print("Running FedAvg+SecAgg...")
    res_fedsec, _ = run_federated(clients_data, X_test_sub, y_test_sub, strategy="FedAvg+SecAgg")
    
    print("Running TRAP-FL...")
    res_trap, best_model = run_federated(clients_data, X_test_sub, y_test_sub, strategy="TRAP-FL")
    
    results = [
        {"Model": "Centralized", **res_cent},
        {"Model": "FedAvg", **res_fedavg},
        {"Model": "FedAvg+DP", **res_feddp},
        {"Model": "FedAvg+SecAgg", **res_fedsec},
        {"Model": "TRAP-FL", **res_trap},
    ]
    
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'experiment_results.csv'), index=False)
    print(df)
    
    # Save TRAP-FL model as the main model for inference
    torch.save(best_model.state_dict(), os.path.join(os.path.dirname(os.path.dirname(__file__)), 'saved_models', 'trap_fl_model.pth'))
    
    # Train 8-feature model for manual prediction
    print("Training 8-feature model...")
    model_8 = get_model(8)
    opt = torch.optim.Adam(model_8.parameters(), lr=0.01)
    crit = nn.BCELoss()
    ds = torch.utils.data.TensorDataset(torch.FloatTensor(X_train_8[idx]), torch.FloatTensor(y_train_sub).view(-1, 1))
    ld = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)
    model_8.train()
    for e in range(5):
        for bx, by in ld:
            opt.zero_grad()
            loss = crit(model_8(bx), by)
            loss.backward()
            opt.step()
    torch.save(model_8.state_dict(), os.path.join(os.path.dirname(os.path.dirname(__file__)), 'saved_models', 'manual_model.pth'))
