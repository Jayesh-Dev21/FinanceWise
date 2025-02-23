# FinanceWise 🏦
 
It provides :
- Fraud Detection Services (Sentinel)
- Financial Guidance (Moneto)
- Portfolio Management (Future Project)

# FinanceWise :bank:

**Next-Generation Financial Intelligence Platform**  
🌐 Live Demo: [https://financewise-ldv2.onrender.com/](https://financewise-ldv2.onrender.com/)  
📄 Presentation: [View Slides](https://www.canva.com/design/DAGf2exzPcI/_HfoIqdbJeR1s-g6T3n7IQ/edit?utm_content=DAGf2exzPcI&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)

[![Flask](https://img.shields.io/badge/Flask-3.0.2-important)](https://flask.palletsprojects.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-red)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.40.0-yellow)](https://huggingface.co/docs/transformers)

![main](https://github.com/user-attachments/assets/7d28efe1-c7b2-4aca-8a2c-5cbd66012102)


## Overview 💡
FinanceWise is an AI-powered financial platform combining two robust systems:
1. **Sentinel** 🛡️ - Advanced Fraud Detection System
2. **Moneto** 💬 - Intelligent Financial Advisor


## Key Features ✨
### Sentinel Fraud Detection System
```json
{
  "Real-time transaction analysis",
  "Deep learning model with 99.9% accuracy",
  "Suspicious pattern recognition",
  "AI-generated risk explanations"
}
```
![sentinel](https://github.com/user-attachments/assets/78d2bc40-9d81-43b1-87a9-57a30adf2ac5)

### Moneto Financial Advisor
```json
{
  "Llama-3 70B-powered financial guidance",
  "Real-time market insights",
  "Personalized wealth management",
  "Risk assessment strategies",
}
```
![moneto](https://github.com/user-attachments/assets/2ccf4ac1-31e8-4ed6-a664-817424aaef02)

### Vigilant Portfolio Manager (Future plans)
```json
{

  "*** COMING SOON ***"

}
```

## Tech Stack 🛠️
**Backend**  
![Flask](https://img.shields.io/badge/-Flask-000000?style=flat&logo=flask)

**Frontend**  
![HTML5](https://img.shields.io/badge/-HTML5-E34F26?style=flat&logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/-CSS3-1572B6?style=flat&logo=css3)
![JavaScript](https://img.shields.io/badge/-JavaScript-F7DF1E?style=flat&logo=javascript&logoColor=black)

**APIs & Services**  
![Groq Cloud Service](https://img.shields.io/badge/-Groq-00FF00?style=flat)
![Llama3](https://img.shields.io/badge/-Llama3-7B3F00?style=flat)

**Personal Model**  
 SENTINAL : Self-Created and Self Trained


## Prerequisites

- Python 3.9+
- pip package manager
- Git version control

## Step-by-Step Setup

1. **Clone repository:**
   ```bash
   git clone https://github.com/Jayesh-Dev21/FinanceWise.git
   cd FinanceWise
   ```
2. **Create virtual environment:**
   ```bash
   python -m venv venv
   # Linux/MacOS
   source venv/bin/activate
   # Windows
   venv\Scripts\activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Configure environment variables:**
   Create a `.env` file in the root directory:
   ```env
   GROQ_API_KEY="your_groq_api_key_here"
   FLASK_DEBUG=0
   PORT=1200
   ```
5. **Download ML models:**
   ```bash
   # Place scaler.pkl and fraud_detection_model.pth in root directory
   ```

## Usage Guide 📖

### Running the Application
```bash
python app.py
```

### Accessing the Web Interface
Open your browser and navigate to:
- **Home Page**: [http://localhost:1200](http://localhost:1200)
- **Sentinel Dashboard**: [http://localhost:1200/fraud_detection](http://localhost:1200/fraud_detection)
- **Moneto Interface**: [http://localhost:1200/financial_advice](http://localhost:1200/financial_advice)

## API Documentation 🔌

### Endpoints

| Endpoint   | Method | Description                  | Parameters  |
|------------|--------|------------------------------|-------------|
| `/predict` | POST   | Fraud detection analysis  | JSON transaction data |
| `/chat`    | POST   | Financial advice generation | `{ message: string }` |

### Sample Inputs

#### Fraud Detection (POST `/predict`):
```json
{
  "amount": 1500,
  "credit_limit": 5000,
  "use_chip": "Chip Transaction",
  "merchant_category": "Online Retail",
  "transaction_date": "2024-03-15"
}
```

#### Financial Advice (POST `/chat`):
```json
{
  "message": "Should I invest 20% of my salary in cryptocurrency?"
}
```

### Response Formats

#### Successful Fraud Detection Response:
```json
{
  "probability": 0.92,
  "prediction": "Fraud",
  "explanation": "Risk assessment: 92.3% suspicious activity likelihood"
}
```

#### Financial Advice Response:
```json
{
  "response": "While cryptocurrency can offer high returns, I recommend..."
}
```

## Project Structure 📂

```
FinanceWise/
├── public/                    # Frontend assets
│   ├── static/                # Static resources
│   │   ├── css/               # Stylesheets
│   │   ├── js/                # Client-side scripts
│   │   └── images/            # Visual assets
│   └── *.html                 # Application views
├── app.py                     # Flask application
├── requirements.txt           # Dependency list
├── scaler.pkl                 # Feature scaler
├── fraud_detection_model.pth  # Trained model
└── .env                       # Environment configuration
```

## Contributors 👥

- **Jaydeep Pokahriya** – Full Web Stack Development (Flask Backend + Frontend) and GitHub Handling
- **Harsh** – Sentinel AI Model Training & Optimization
- **Jayesh** – Moneto API Integration & LLM Configuration in Sentinel
- **Krrish** – Project Presentation & Technical Documentation


## Acknowledgments 🙏

- **Groq**: For ultra-low latency LLM inference
- **Hugging Face**: For transformer models (pre-trained models)
- **Render**: For reliable cloud hosting
- **PyTorch Community**: For deep learning framework

## ⚠️ Important Notes:

- Obtain a Groq API key from Groq Cloud.
- Do not commit the `.env` file to version control.
- For production deployment, set `FLASK_DEBUG=0`.






