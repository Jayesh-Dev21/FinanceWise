import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import kagglehub
import glob
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pickle

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Download dataset
print("Downloading dataset...")
path = kagglehub.dataset_download("computingvictor/transactions-fraud-datasets")
print(f"Path to dataset files: {path}")

# Load all CSV files with error handling
csv_files = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
if not csv_files:
    raise FileNotFoundError("No CSV files found in the dataset path")

data_dict = {}
for file in csv_files:
    try:
        filename = os.path.basename(file).split('.')[0]
        data_dict[filename] = pd.read_csv(file)
        print(f"Loaded {filename} with shape: {data_dict[filename].shape}")
    except Exception as e:
        print(f"Error loading {file}: {e}")

# Extract datasets with validation
required_datasets = ['transactions_data', 'cards_data', 'users_data']
for dataset in required_datasets:
    if dataset not in data_dict:
        raise ValueError(f"Missing required dataset: {dataset}")

transactions_df = data_dict['transactions_data']
cards_df = data_dict['cards_data']
users_df = data_dict['users_data']

# Print column names for debugging
print("Transactions columns:", transactions_df.columns.tolist())
print("Cards columns:", cards_df.columns.tolist())
print("Users columns:", users_df.columns.tolist())

# Data preprocessing
def preprocess_data(cards_df, transactions_df):
    """Preprocess numerical columns and handle missing values"""
    if 'credit_limit' in cards_df.columns:
        cards_df['credit_limit'] = pd.to_numeric(
            cards_df['credit_limit'].replace(r'[\$,]', '', regex=True), 
            errors='coerce'
        )
    if 'amount' in transactions_df.columns:
        transactions_df['amount'] = pd.to_numeric(
            transactions_df['amount'].replace(r'[\$,]', '', regex=True), 
            errors='coerce'
        )
    return cards_df, transactions_df

cards_df, transactions_df = preprocess_data(cards_df, transactions_df)

# Generate fraud indicators with column validation
def generate_fraud_indicators(transactions_df, cards_df, users_df):
    print("Generating fraud indicators...")
    
    # Validate required columns for merging
    required_transaction_cols = ['card_id']
    required_cards_cols = ['id']
    required_users_cols = ['id']
    
    for col in required_transaction_cols:
        if col not in transactions_df.columns:
            raise ValueError(f"Required column '{col}' not found in transactions_df")
    for col in required_cards_cols:
        if col not in cards_df.columns:
            raise ValueError(f"Required column '{col}' not found in cards_df")
    for col in required_users_cols:
        if col not in users_df.columns:
            raise ValueError(f"Required column '{col}' not found in users_df")

    # First merge with cards, keeping track of original client_id
    df = transactions_df.merge(cards_df, left_on='card_id', right_on='id', how='left', suffixes=('_trans', '_card'))
    
    # Use the transaction's client_id (which should now be 'client_id_trans' after merge)
    client_col = 'client_id_trans'
    if client_col not in df.columns:
        # Fallback to original client_id if suffix didn't work as expected
        possible_client_cols = ['client_id', 'client_id_trans', 'client_id_x', 'client_id_y']
        for col in possible_client_cols:
            if col in df.columns:
                client_col = col
                break
        else:
            print("Warning: No client identifier found. Proceeding without user data merge")
            client_col = None
    
    if client_col:
        print(f"Using '{client_col}' as client identifier")
        df = df.merge(users_df, left_on=client_col, right_on='id', how='left')
    
    # Enhanced fraud indicators
    df['amount_ratio'] = df['amount'] / df['credit_limit'].replace(0, np.nan)
    
    # Check for chip-related columns
    if 'has_chip' in df.columns and 'use_chip' in df.columns:
        df['suspicious_no_chip'] = ((df['has_chip'] == 'YES') & (df['use_chip'] == 'NO')).astype(int)
    else:
        df['suspicious_no_chip'] = 0
        print("Warning: Chip-related columns not found")

    if 'zip' in df.columns:
        df['zip_first3'] = df['zip'].astype(str).str[:3].fillna('000')
    else:
        df['zip_first3'] = '000'
        print("Warning: 'zip' column not found")
    
    # Fraud detection with available columns
    conditions = []
    if 'amount_ratio' in df.columns:
        conditions.append(df['amount_ratio'] > 0.8)
    if 'suspicious_no_chip' in df.columns:
        conditions.append(df['suspicious_no_chip'] == 1)
    if 'amount' in df.columns and 'credit_limit' in df.columns:
        conditions.append(df['amount'] > df['credit_limit'].fillna(float('inf')))
    
    df['is_fraud'] = np.any(conditions, axis=0).astype(int) if conditions else 0
    
    # Clean up merged column names
    columns_to_drop = [col for col in df.columns if col.endswith(('_x', '_y')) or col.startswith('id_')]
    return df.drop(columns_to_drop, axis=1, errors='ignore')

df = generate_fraud_indicators(transactions_df, cards_df, users_df)
print(f"Generated dataset with shape: {df.shape}")
print(f"Fraud rate: {df['is_fraud'].mean():.2%}")

# LLM initialization
print("Initializing LLM for fraud explanation...")
try:
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    llm_model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )
    use_llm = True
except Exception as e:
    print(f"Could not initialize LLM: {e}")
    use_llm = False

def generate_fraud_explanation(transaction_data):
    if not use_llm:
        return "LLM not available"
    
    transaction_text = (
        f"Amount: ${transaction_data['amount']:.2f}, "
        f"Credit Limit: ${transaction_data['credit_limit']:.2f}, "
        f"Ratio: {transaction_data['amount_ratio']:.2f}, "
        f"Chip Usage: {transaction_data.get('use_chip', 'N/A')}"
    )
    
    inputs = tokenizer(transaction_text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = llm_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return f"Fraud probability: {probs[0][1]:.2f}"

# Data preparation
categorical_cols = df.select_dtypes(include=['object']).columns
df[categorical_cols] = df[categorical_cols].apply(lambda x: pd.factorize(x)[0])

X = df.drop(['is_fraud'] + [col for col in df.columns if 'id' in col.lower() or 'date' in col.lower()], 
           axis=1, 
           errors='ignore')
y = df['is_fraud']

# Handle missing values
X = X.fillna(X.mean())
valid_mask = X.notna().all(axis=1)
X, y = X[valid_mask], y[valid_mask]

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Custom Dataset
class FraudDataset(Dataset):
    def __init__(self, features, targets):
        assert len(features) == len(targets), "Features and targets length mismatch"
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets.values).reshape(-1, 1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# Neural Network
class FraudDetectionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return torch.sigmoid(self.network(x))

# Training function
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=10):
    best_auc = 0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        y_pred, y_true = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                y_pred.extend(outputs.numpy().flatten())
                y_true.extend(labels.numpy().flatten())
        
        auc = roc_auc_score(y_true, y_pred)
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Loss: {train_loss/len(train_loader):.4f}, "
              f"AUC: {auc:.4f}")
        
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), "best_fraud_model.pth")

# Execution
train_dataset = FraudDataset(X_train_scaled, y_train)
test_dataset = FraudDataset(X_test_scaled, y_test)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

model = FraudDetectionModel(X_train_scaled.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

train_model(model, train_loader, test_loader, criterion, optimizer)

# Final evaluation
model.eval()
y_pred = []
with torch.no_grad():
    for inputs, _ in test_loader:
        y_pred.extend(model(inputs).numpy().flatten())
print("\nClassification Report:")
print(classification_report(y_test, (np.array(y_pred) > 0.5).astype(int)))

# Save final model
torch.save(model.state_dict(), "fraud_detection_model.pth")
print("Model saved successfully!")
