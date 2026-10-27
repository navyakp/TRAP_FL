import os
import pandas as pd
import numpy as np
import urllib.request
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'saved_models')

TRAIN_URL = "https://raw.githubusercontent.com/Nir-J/ML-Projects/master/UNSW-Network_Packet_Classification/UNSW_NB15_training-set.csv"
TEST_URL = "https://raw.githubusercontent.com/Nir-J/ML-Projects/master/UNSW-Network_Packet_Classification/UNSW_NB15_testing-set.csv"

# The 8 features selected for manual prediction
TOP_8_FEATURES = ['sttl', 'ct_state_ttl', 'sload', 'rate', 'sbytes', 'dbytes', 'dload', 'smean']

def ensure_data_exists():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    train_path = os.path.join(DATA_DIR, 'UNSW_NB15_training-set.csv')
    test_path = os.path.join(DATA_DIR, 'UNSW_NB15_testing-set.csv')
    
    if not os.path.exists(train_path):
        print("Downloading training set...")
        try:
            urllib.request.urlretrieve(TRAIN_URL, train_path)
        except Exception as e:
            print(f"Failed to download from {TRAIN_URL}. Generating dummy data for testing...")
            _generate_dummy_data(train_path, test_path)
            return train_path, test_path
            
    if not os.path.exists(test_path):
        print("Downloading testing set...")
        try:
            urllib.request.urlretrieve(TEST_URL, test_path)
        except Exception as e:
            print(f"Failed to download from {TEST_URL}.")
            
    return train_path, test_path

def _generate_dummy_data(train_path, test_path):
    # Dummy data generator if URLs are broken
    cols = ['id'] + TOP_8_FEATURES + ['proto', 'service', 'state', 'label', 'attack_cat']
    df = pd.DataFrame(np.random.randn(1000, len(cols)), columns=cols)
    df['id'] = range(1000)
    df['proto'] = np.random.choice(['tcp', 'udp'], 1000)
    df['service'] = np.random.choice(['http', 'ftp', '-'], 1000)
    df['state'] = np.random.choice(['FIN', 'INT', 'CON'], 1000)
    df['attack_cat'] = np.random.choice(['Normal', 'Exploits'], 1000)
    df['label'] = (df['attack_cat'] != 'Normal').astype(int)
    df.to_csv(train_path, index=False)
    df.to_csv(test_path, index=False)

def preprocess_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Drop id and attack_cat (we only predict binary 'label')
    cols_to_drop = ['id', 'attack_cat']
    train_df.drop(columns=[c for c in cols_to_drop if c in train_df.columns], inplace=True)
    test_df.drop(columns=[c for c in cols_to_drop if c in test_df.columns], inplace=True)
    
    # Identify categorical columns
    categorical_cols = train_df.select_dtypes(include=['object']).columns
    
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        # Fit on both train and test to avoid unseen labels
        all_data = pd.concat([train_df[col], test_df[col]]).astype(str)
        le.fit(all_data)
        train_df[col] = le.transform(train_df[col].astype(str))
        test_df[col] = le.transform(test_df[col].astype(str))
        label_encoders[col] = le
        
    # Scale numeric features
    X_train = train_df.drop(columns=['label'])
    y_train = train_df['label'].values
    X_test = test_df.drop(columns=['label'])
    y_test = test_df['label'].values
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Extract 8-feature dataset for manual prediction
    feat_indices = [X_train.columns.get_loc(c) for c in TOP_8_FEATURES if c in X_train.columns]
    X_train_8 = X_train_scaled[:, feat_indices]
    X_test_8 = X_test_scaled[:, feat_indices]
    
    scaler_8 = StandardScaler()
    scaler_8.fit(X_train[TOP_8_FEATURES]) # separate scaler just for 8 features for manual entry
    
    # Save scalers and encoders
    with open(os.path.join(MODEL_DIR, 'scaler_full.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(MODEL_DIR, 'scaler_8.pkl'), 'wb') as f:
        pickle.dump(scaler_8, f)
    with open(os.path.join(MODEL_DIR, 'label_encoders.pkl'), 'wb') as f:
        pickle.dump(label_encoders, f)
    with open(os.path.join(MODEL_DIR, 'features_full.pkl'), 'wb') as f:
        pickle.dump(X_train.columns.tolist(), f)
        
    return (X_train_scaled, y_train, X_test_scaled, y_test), (X_train_8, y_train, X_test_8, y_test)

def create_non_iid_splits(X, y, num_clients=3, alpha=0.5):
    # Dirichlet distribution for non-IID split
    # For binary classification
    min_size = 0
    K = 2
    N = len(y)
    net_dataidx_map = {}
    
    while min_size < 10:
        idx_batch = [[] for _ in range(num_clients)]
        for k in range(K):
            idx_k = np.where(y == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])
            
    client_data = []
    for j in range(num_clients):
        np.random.shuffle(idx_batch[j])
        client_data.append((X[idx_batch[j]], y[idx_batch[j]]))
        
    return client_data

if __name__ == "__main__":
    train_p, test_p = ensure_data_exists()
    full_data, small_data = preprocess_data(train_p, test_p)
    print("Full data shape:", full_data[0].shape)
    print("8-feature data shape:", small_data[0].shape)
    
    clients_data = create_non_iid_splits(full_data[0], full_data[1], num_clients=3)
    for i, (cx, cy) in enumerate(clients_data):
        print(f"Client {i} size: {len(cy)}, positives: {sum(cy)}")
