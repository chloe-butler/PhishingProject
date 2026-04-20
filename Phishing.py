import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("phishing_legit_dataset_KD_10000.csv")
df = df.dropna()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

df['clean_text'] = df['text'].apply(clean_text)

X = df['clean_text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

lr_clf = LogisticRegression(max_iter=1000, class_weight='balanced')
lr_clf.fit(X_train_vec, y_train)

y_pred_lr = lr_clf.predict(X_test_vec)
print("=== Logistic Regression ===")
print(classification_report(y_test, y_pred_lr))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))

new_emails = [
    "Your account has been locked. Click the link to verify your credentials immediately.",
    "Hi team, the meeting is at 10 AM tomorrow. See you then!",
    "You won a gift card! Provide your credit card to claim your prize.",
    "Don’t forget the report is due next Monday. Please send it by EOD.",
    "Can we reschedule lunch for tomorrow afternoon?"
]

new_clean = [clean_text(e) for e in new_emails]
new_vec = vectorizer.transform(new_clean)
predictions_lr = lr_clf.predict(new_vec)

print("\nNew email predictions:")
for email, label in zip(new_emails, predictions_lr):
    print(f"{label}: {email}")