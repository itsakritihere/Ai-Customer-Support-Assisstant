import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the training dataset
df = pd.read_csv("banking77_train.csv")

print("Dataset shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nNumber of categories:")
print(df["label"].nunique())

print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nSample tickets:")
for i in range(5):
    print(f"\nTicket: {df.loc[i, 'text']}")
    print(f"Label: {df.loc[i, 'label']}")

# Separate input and target
X = df["text"]
y = df["label"]

print("\nInput X:")
print(X.head())

print("\nTarget y:")
print(y.head())
train_df = pd.read_csv("banking77_train.csv")
test_df = pd.read_csv("banking77_test.csv")

X_train = train_df["text"]
y_train = train_df["label"]

X_test = test_df["text"]
y_test = test_df["label"]

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))
# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

# Learn vocabulary from training data and transform it
X_train_tfidf = vectorizer.fit_transform(X_train)

# Only transform test data
X_test_tfidf = vectorizer.transform(X_test)

print("\nTF-IDF training shape:", X_train_tfidf.shape)
print("TF-IDF testing shape:", X_test_tfidf.shape)
# Create Logistic Regression model
model = LogisticRegression(
    max_iter=1000
)

# Train the model
model.fit(X_train_tfidf, y_train)
# Create models directory
os.makedirs("models", exist_ok=True)

# Save the trained model
joblib.dump(model, "models/logistic_regression.pkl")

# Save the TF-IDF vectorizer
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("\nModel and vectorizer saved successfully!")

print("\nModel training completed!")
# Make predictions on unseen test data
y_pred = model.predict(X_test_tfidf)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)
print("Model Accuracy (%):", accuracy * 100)

# Detailed evaluation
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\n==============================")
print("LABEL EXAMPLES")
print("==============================")

for label in sorted(train_df["label"].unique()):
    print(f"\nLABEL {label}")
    print(train_df[train_df["label"] == label]["text"].iloc[0])