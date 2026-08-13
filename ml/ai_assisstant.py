import os
import joblib
from categories import CATEGORY_NAMES

from dotenv import load_dotenv
from groq import Groq


# ==========================================
# 1. Load environment variables
# ==========================================

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env")


# ==========================================
# 2. Load ML model and TF-IDF vectorizer
# ==========================================

model = joblib.load(
    "models/logistic_regression.pkl"
)

vectorizer = joblib.load(
    "models/tfidf_vectorizer.pkl"
)


# ==========================================
# 3. Temporary category mapping
# ==========================================

# We will replace this with the complete
# 77-category mapping later.



# ==========================================
# 4. Create Groq client
# ==========================================

client = Groq(
    api_key=api_key
)


# ==========================================
# 5. Get customer ticket
# ==========================================

ticket = input("\nEnter customer ticket: ")


# ==========================================
# 6. ML prediction
# ==========================================

ticket_tfidf = vectorizer.transform([ticket])

prediction = model.predict(ticket_tfidf)[0]

probabilities = model.predict_proba(ticket_tfidf)[0]

confidence = max(probabilities)


# ==========================================
# 7. Convert label to category
# ==========================================

category = CATEGORY_NAMES.get(
    prediction,
    f"unknown_category_{prediction}"
)


# ==========================================
# 8. Create prompt for Groq
# ==========================================

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
- Do not claim to access customer accounts, orders, tracking systems,
  payment systems, or internal company systems.
- Do not ask for passwords, PINs, OTPs, CVV, or full card numbers.
- If account-specific investigation is required, tell the customer
  to contact the company's official support team.
- Do not mention the machine learning model, confidence score, or Groq.
- Keep the response concise, around 3-5 sentences.
"""


# ==========================================
# 9. Generate AI response using Groq
# ==========================================

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful and professional customer support assistant."
        },
        {
            "role": "user",
            "content": prompt
        }
    ],
    temperature=0.3
)


# ==========================================
# 10. Get generated response
# ==========================================

ai_response = response.choices[0].message.content


# ==========================================
# 11. Display complete result
# ==========================================

print("\n========================================")
print("      AI CUSTOMER SUPPORT ASSISTANT")
print("========================================")

print("\nCustomer Ticket:")
print(ticket)

print("\nML Classification:")
print("Predicted Label:", prediction)
print("Predicted Category:", category)
print("Confidence:", round(confidence * 100, 2), "%")

print("\nAI Generated Response:")
print(ai_response)

print("\n========================================")