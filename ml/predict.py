import joblib


# --------------------------------
# 1. Load saved ML components
# --------------------------------

model = joblib.load("models/logistic_regression.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")


# --------------------------------
# 2. Category mapping
# --------------------------------

CATEGORY_NAMES = {
    0: "activate_my_card",
    11: "card_arrival"
}


# --------------------------------
# 3. Get customer ticket
# --------------------------------

ticket = input("\nEnter customer ticket: ")


# --------------------------------
# 4. Convert ticket to TF-IDF
# --------------------------------

ticket_tfidf = vectorizer.transform([ticket])


# --------------------------------
# 5. Predict category
# --------------------------------

prediction = model.predict(ticket_tfidf)[0]


# --------------------------------
# 6. Calculate confidence
# --------------------------------

probabilities = model.predict_proba(ticket_tfidf)[0]

confidence = max(probabilities)


# --------------------------------
# 7. Convert label to category
# --------------------------------

category = CATEGORY_NAMES.get(
    prediction,
    f"unknown_category_{prediction}"
)


# --------------------------------
# 8. Display result
# --------------------------------

print("\n--------------------------------")
print("     AI CUSTOMER SUPPORT ASSISTANT")
print("--------------------------------")

print("\nCustomer Ticket:")
print(ticket)

print("\nPredicted Label:", prediction)

print("Predicted Category:", category)

print(
    "Confidence:",
    round(confidence * 100, 2),
    "%"
)

print("--------------------------------")