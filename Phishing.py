
# Phishing.py
# Simple phishing detection prototype using Scikit-learn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1️⃣ Load dataset (replace with your CSV path)
# Example: dataset from Kaggle with 'email_text' and 'label' columns
df = pd.read_csv('emails.csv')  # your dataset file here

# 2️⃣ Preprocessing
df = df.dropna()  # remove missing values

# Features and target
X = df['email_text']        # column containing email text
y = df['label']             # 1 = phishing, 0 = legitimate

# 3️⃣ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4️⃣ Convert text to numerical features
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 5️⃣ Train Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_vec, y_train)

# 6️⃣ Make predictions
y_pred = clf.predict(X_test_vec)

# 7️⃣ Evaluate model
print(classification_report(y_test, y_pred))

# 8️⃣ Predict new emails (example)
new_emails = [
    "Your account has been suspended! Click here to verify.",
    "Meeting tomorrow at 10 AM in the conference room."
]
new_vec = vectorizer.transform(new_emails)
predictions = clf.predict(new_vec)
for email, label in zip(new_emails, predictions):
    print(f"Email: {email}\nPrediction: {'Phishing' if label == 1 else 'Legitimate'}\n")


# Test for Syd