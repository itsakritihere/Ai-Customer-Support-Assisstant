from pathlib import Path
import os

import joblib
from dotenv import load_dotenv
from groq import Groq
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ==========================================
# 1. Project paths
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "ml" / "models" / "logistic_regression.pkl"
VECTORIZER_PATH = BASE_DIR / "ml" / "models" / "tfidf_vectorizer.pkl"
CATEGORIES_PATH = BASE_DIR / "ml" / "categories.py"


# ==========================================
# 2. Load ML model and TF-IDF vectorizer
# ==========================================

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


# ==========================================
# 3. Load category mapping
# ==========================================

import sys

sys.path.insert(0, str(BASE_DIR / "ml"))

from categories import CATEGORY_NAMES


# ==========================================
# 4. Load environment variables
# ==========================================

load_dotenv(BASE_DIR / "backend" / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found")


# ==========================================
# 5. Create Groq client
# ==========================================

client = Groq(api_key=GROQ_API_KEY)


# ==========================================
# 6. FastAPI application
# ==========================================

app = FastAPI(
    title="AI Customer Support Assistant",
    description="ML-powered customer support ticket classification API",
    version="1.0.0"
)


# ==========================================
# 7. CORS
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        
        # Add your Vercel frontend URL here later
        # "https://your-frontend.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# 8. Request structure
# ==========================================

class TicketRequest(BaseModel):
    ticket: str


# ==========================================
# 9. Home endpoint
# ==========================================

@app.get("/")
def home():
    return {
        "message": "AI Customer Support Assistant API is running"
    }


# ==========================================
# 10. Prediction endpoint
# ==========================================

@app.post("/predict")
def predict_ticket(request: TicketRequest):

    ticket = request.ticket.strip()

    if not ticket:
        return {
            "error": "Ticket cannot be empty"
        }

    # ======================================
    # Convert ticket to TF-IDF
    # ======================================

    ticket_tfidf = vectorizer.transform([ticket])


    # ======================================
    # ML prediction
    # ======================================

    prediction = model.predict(ticket_tfidf)[0]

    probabilities = model.predict_proba(ticket_tfidf)[0]

    confidence = max(probabilities)


    # ======================================
    # Convert label to category
    # ======================================

    category = CATEGORY_NAMES.get(
        int(prediction),
        f"unknown_category_{prediction}"
    )


    # ======================================
    # Generate AI response
    # ======================================

    prompt = f"""
You are an AI customer support assistant.

Customer ticket:
{ticket}

Detected support category:
{category}

Generate a helpful customer support response.

Rules:

- Be polite, professional, and empathetic.
- Directly address the customer's issue.
- Give practical steps the customer can take.
- Do not claim to access customer accounts.
- Do not claim to access orders or tracking systems.
- Do not claim that you performed an action.
- Do not ask for passwords, PINs, OTPs, CVV, or full card numbers.
- If account-specific investigation is required, tell the customer
  to contact the official support team.
- Do not mention the ML model, confidence score, or Groq.
- Keep the response concise, around 3-5 sentences.
"""


    # ======================================
    # Groq API
    # ======================================

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",

        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful and professional "
                    "customer support assistant."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.3
    )


    ai_response = response.choices[0].message.content


    # ======================================
    # Return result
    # ======================================

    return {
        "ticket": ticket,
        "predicted_label": int(prediction),
        "category": category,
        "confidence": round(
            float(confidence) * 100,
            2
        ),
        "ai_response": ai_response
    }