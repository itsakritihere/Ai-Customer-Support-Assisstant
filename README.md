# 🤖 AI Customer Support Assistant

An AI-powered customer support automation system that combines **Natural Language Processing (NLP), Machine Learning, Generative AI, FastAPI, and React** to classify customer support tickets and generate context-aware responses.

The system uses **TF-IDF + Logistic Regression** to classify customer queries into **77 support categories** and then uses a **Groq-powered LLM** to generate a helpful response based on the detected category.

---

## 🚀 Features

- 🎫 Real-time customer ticket classification
- 🧠 NLP-based text classification using TF-IDF
- 🤖 Logistic Regression machine learning model
- 📊 Classification across 77 customer-support categories
- 📈 Confidence score for predictions
- 📝 Precision, Recall and F1-score evaluation
- 💬 AI-generated customer support responses
- ⚡ FastAPI backend
- ⚛️ React frontend
- 🔐 Environment-variable based API key management
- 🌐 REST API architecture
- ❌ No database required

---

## 🏗️ System Architecture

```text
                    Customer
                       │
                       ▼
              React Frontend
                       │
                       ▼
              FastAPI Backend
                       │
              ┌────────┴────────┐
              ▼                 ▼
       ML Classifier          Groq API
       TF-IDF +               Generative
       Logistic               AI Response
       Regression
              │                 │
              └────────┬────────┘
                       ▼
                Final Response
                       │
                       ▼
                React Dashboard
                Machine Learning Pipeline

The ticket classification pipeline works as follows:

Customer Ticket
       │
       ▼
Text Preprocessing
       │
       ▼
TF-IDF Vectorization
       │
       ▼
Logistic Regression
       │
       ▼
Predicted Category
       │
       ▼
Confidence Score

The predicted category is then passed to the Generative AI component:

Predicted Category
       +
Customer Ticket
       │
       ▼
    Groq LLM
       │
       ▼
AI-generated Support Response
📊 Dataset

This project uses the Banking77 dataset.

The dataset contains:

13,083 total samples
10,003 training samples
3,080 testing samples
77 customer-support categories

Example categories include:

card_arrival
card_linking
exchange_rate
card_payment_wrong_exchange_rate
card_not_working
cash_withdrawal_not_recognised
pending_card_payment
cash_withdrawal_charge
lost_or_stolen_card
top_up_failed
verify_my_identity
country_support
📈 Model Performance

The initial Logistic Regression model achieved:

Metric	Score
Accuracy	83.02%
Macro Precision	0.84
Macro Recall	0.83
Macro F1-score	0.83
Weighted F1-score	0.83

The model was evaluated on 3,080 unseen test samples.

🔍 Example
Customer Ticket
My card hasn't arrived yet and I have been waiting for two weeks.
ML Classification
Predicted Label: 11
Predicted Category: card_arrival
Confidence: 74.14%
AI Generated Response
Dear customer,

I apologize for the delay in receiving your card and appreciate
your patience. I understand how frustrating it can be to wait
for an important item. Please check your card delivery tracking
information or contact customer support for further assistance.
🛠️ Tech Stack
Machine Learning
Python
Pandas
NumPy
Scikit-learn
TF-IDF
Logistic Regression
Joblib
Generative AI
Groq API
Large Language Model (LLM)
Prompt-based response generation
Backend
FastAPI
Uvicorn
Python
REST API
Pydantic
CORS
Frontend
React.js
Vite
JavaScript
HTML
CSS
📁 Project Structure
AI-Customer-Support-Assistant/
│
├── ml/
│   ├── models/
│   │   ├── logistic_regression.pkl
│   │   └── tfidf_vectorizer.pkl
│   │
│   ├── banking77_train.csv
│   ├── banking77_test.csv
│   ├── train.py
│   ├── predict.py
│   ├── ai_assistant.py
│   ├── download_dataset.py
│   └── requirements.txt
│
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   └── ...
│
├── frontend/
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── ...
│
├── .gitignore
└── README.md
⚙️ Installation
1. Clone the repository
git clone https://github.com/YOUR_USERNAME/AI-Customer-Support-Assistant.git
cd AI-Customer-Support-Assistant
2. Create Python Virtual Environment
cd ml
python -m venv venv
Windows
venv\Scripts\activate
macOS/Linux
source venv/bin/activate
3. Install Python Dependencies
pip install -r requirements.txt
🤖 Train the ML Model

Run:

python train.py

The training process:

Loads the Banking77 dataset
Separates text and labels
Converts text into TF-IDF features
Trains Logistic Regression
Evaluates the model
Saves the trained model and vectorizer

Generated files:

models/
├── logistic_regression.pkl
└── tfidf_vectorizer.pkl
🔑 Environment Variables

Create a .env file.

GROQ_API_KEY=your_groq_api_key

Never commit your .env file to GitHub.

Add this to .gitignore:

.env
venv/
__pycache__/
*.pyc
⚡ Running the Backend

Navigate to the backend:

cd backend

Start FastAPI:

uvicorn main:app --reload

The API will be available at:

http://127.0.0.1:8000

FastAPI documentation:

http://127.0.0.1:8000/docs
⚛️ Running the Frontend

Navigate to the frontend:

cd frontend

Install dependencies:

npm install

Start the development server:

npm run dev

The frontend will be available at the URL shown by Vite.

🔌 API Example
Endpoint
POST /predict
Request
{
  "ticket": "My card hasn't arrived yet"
}
Response
{
  "category": "card_arrival",
  "confidence": 0.7414,
  "response": "I apologize for the delay..."
}
🔐 Security

API keys are stored using environment variables.

Example:

GROQ_API_KEY=your_api_key

Sensitive files such as .env and the Python virtual environment should not be committed to GitHub.

🎯 Future Improvements
 Improve classification accuracy through hyperparameter tuning
 Add Top-3 predicted categories
 Add confidence-based fallback responses
 Add conversation history
 Add authentication
 Add automated testing
 Improve frontend UI/UX
 Deploy FastAPI backend
 Deploy React frontend
 Add monitoring and logging
📚 Learning Outcomes

This project demonstrates practical experience with:

Natural Language Processing
Text preprocessing
TF-IDF feature extraction
Supervised Machine Learning
Multiclass classification
Logistic Regression
Model evaluation
REST API development
FastAPI
React
Generative AI
Prompt engineering
API integration
Full-stack application development
👩‍💻 Author

Akriti Chauhan

B.Tech — Electronics & Communication Engineering

Interested in:

Software Engineering
Full-Stack Development
Java Backend Development
Artificial Intelligence & Machine Learning