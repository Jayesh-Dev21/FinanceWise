import os
import pickle
import torch
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from torch.utils.data import DataLoader
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

# Configure Flask app
app = Flask(__name__, 
          static_folder='public/static', 
          template_folder='public')
CORS(app)

# Configuration
port = int(os.environ.get('PORT', 1200))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize fraud detection components
print("\nInitializing fraud detection system...")
try:
    # Load scaler
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    
    # Load fraud detection model
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

    model = FraudDetectionModel(scaler.n_features_in_).to(device)
    model.load_state_dict(torch.load("fraud_detection_model.pth", map_location=device))
    model.eval()
    print("✅ Fraud detection model loaded successfully!")

except Exception as e:
    print(f"❌ Fraud detection initialization failed: {e}")
    model = None

# Initialize LLM components
print("\nInitializing AI components...")
try:
    # Financial advice LLM
    groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    
    # Fraud explanation model
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    llm_model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased-sst-2-english"
    ).to(device)
    print("✅ AI models loaded successfully!")
except Exception as e:
    print(f"❌ AI initialization failed: {e}")
    # groq_client = None
    llm_model = None

# Helper functions
def preprocess_input(data):
    """Process transaction data for fraud detection"""
    df = pd.DataFrame([data])
    
    # Clean and prepare data
    categorical_cols = df.select_dtypes(include=['object']).columns
    df[categorical_cols] = df[categorical_cols].apply(lambda x: pd.factorize(x)[0])
    
    # Handle missing features
    expected_features = scaler.feature_names_in_
    X = df.drop(columns=[col for col in df.columns if 'id' in col.lower() or 'date' in col.lower()], errors='ignore')
    missing_cols = set(expected_features) - set(X.columns)
    
    for col in missing_cols:
        X[col] = 0  # Default value for missing features
        
    X = X[expected_features]  # Ensure correct order
    return torch.FloatTensor(scaler.transform(X)).to(device)

def generate_explanation(transaction):
    """Generate fraud explanation using LLM"""
    if not llm_model:
        return "Explanation service unavailable"
    
    text = f"Amount: ${transaction['amount']:.2f}, Credit Limit: ${transaction['credit_limit']:.2f}, " \
           f"Ratio: {transaction['amount_ratio']:.2f}, Chip Usage: {transaction.get('use_chip', 'N/A')}"
    
    inputs = tokenizer([text], return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = llm_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    return f"Risk assessment: {probs[0][1]*100:.1f}% suspicious activity likelihood"

# API Endpoints
@app.route('/')
def serve_index():
    return send_from_directory('public', 'index.html')

@app.route('/fraud_detection')
def serve_fraud_detection():
    return send_from_directory('public', 'fraud_detection.html')

@app.route('/financial_advice')
def serve_financial_advice():
    return send_from_directory('public', 'financial_advice.html')

@app.route('/predict', methods=['POST'])
def predict_fraud():
    """Fraud detection prediction endpoint"""
    if not model:
        return jsonify({"error": "Fraud detection system unavailable"}), 503
    
    try:
        transaction = request.json
        X = preprocess_input(transaction)
        
        with torch.no_grad():
            prediction = model(X).cpu().numpy().flatten()[0]
        
        response = {
            "probability": float(prediction),
            "prediction": "Fraud" if prediction > 0.5 else "Not Fraud"
        }
        
        if llm_model:
            response["explanation"] = generate_explanation(transaction)
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/chat', methods=['POST'])
def financial_chat():
    """Financial advice chat endpoint"""
    if not groq_client:
        return jsonify({"error": "Chat service unavailable"}), 503
    
    try:
        user_input = request.json['message']
        
        response = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                         "You are FinanceWise AI,named Moneto, a financial expert who evaluates user decisions. "
            "If the question is finance-related, provide a direct, concise answer, "
            "stating whether the user's thinking is correct and guiding them if necessary. "
            "For non-financial questions, creatively relate them to finance and answer briefly. "
            "Always end non-financial answers(dont write this in very general topics ) with: 'I am not an expert in that field of interest.Lets talk about finance :) '"
            "Always give short and crisp answers which are to the point."
            "Be responsible for other's hard earned money"

                    )
                },
                {"role": "user", "content": user_input}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            top_p=0.7
        )
        
        return jsonify({'response': response.choices[0].message.content})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=os.environ.get('FLASK_DEBUG', False))