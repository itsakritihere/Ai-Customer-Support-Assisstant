
from pydantic import BaseModel
import joblib
import sys
import os

from dotenv import load_dotenv
from groq import Groq
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


# ==========================================
# FastAPI application
# ==========================================


app = FastAPI(
    title="AI Customer Support Assistant",
    description="ML-powered customer support ticket classification API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ==========================================
# Import category mapping
# ==========================================

sys.path.append("../ml")

from categories import CATEGORY_NAMES


# ==========================================
# Load ML model
# ==========================================

model = joblib.load(
    "../ml/models/logistic_regression.pkl"
)

vectorizer = joblib.load(
    "../ml/models/tfidf_vectorizer.pkl"
)


# ==========================================
# Load environment variables
# ==========================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in backend/.env")


# ==========================================
# Create Groq client
# ==========================================

client = Groq(
    api_key=GROQ_API_KEY
)


# ==========================================
# Request structure
# ==========================================

class TicketRequest(BaseModel):
    ticket: str


# ==========================================
# Home endpoint
# ==========================================

@app.get("/")
def home():

    return {
        "message": "AI Customer Support Assistant API is running"
    }


# ==========================================
# Prediction endpoint
# ==========================================

@app.post("/predict")
def predict_ticket(request: TicketRequest):

    ticket = request.ticket.strip()

    if not ticket:

        return {
            "error": "Ticket cannot be empty"
        }


    # ======================================
    # 1. Convert ticket to TF-IDF
    # ======================================

    ticket_tfidf = vectorizer.transform([ticket])


    # ======================================
    # 2. ML prediction
    # ======================================

    prediction = model.predict(ticket_tfidf)[0]

    probabilities = model.predict_proba(ticket_tfidf)[0]

    confidence = max(probabilities)


    # ======================================
    # 3. Convert label to category
    # ======================================

    category = CATEGORY_NAMES.get(
        int(prediction),
        f"unknown_category_{prediction}"
    )


    # ======================================
    # 4. Generate AI response
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
    # 5. Return complete result
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