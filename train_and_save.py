import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# --------------------------
# 1. Load dataset
# --------------------------
data = pd.read_csv("dataset.csv")

# Dataset should have 2 columns: "text" and "label"
X = data["text"]
y = data["label"]

# --------------------------
# 2. Split dataset
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------
# 3. Vectorize text
# --------------------------
vectorizer = TfidfVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# --------------------------
# 4. Train model
# --------------------------
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# --------------------------
# 5. Evaluate model
# --------------------------
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model trained with accuracy: {accuracy:.2f}")

# --------------------------
# 6. Save model & vectorizer
# --------------------------
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ model.pkl and vectorizer.pkl saved successfully!")
