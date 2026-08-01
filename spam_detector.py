import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 1. Expanded Dataset to vastly improve training patterns and accuracy
# In production, swap this out for a full dataset using: pd.read_csv('spam.csv')
data = {
    'message': [
        # --- Ham (Normal Messages) ---
        "Hey, are we still meeting for lunch today?",
        "Can you send me the lecture notes from yesterday's class?",
        "Just wanted to say hello and see how you are doing.",
        "Don't forget to buy milk on your way back home.",
        "Are you free for a call around 4 PM? Let me know.",
        "Sorry, I'm running late for the meeting. Start without me.",
        "Thanks for the birthday wishes! Appreciate it.",
        "The project deadline has been extended to next Friday.",
        "Can we reschedule our tennis match to tomorrow morning?",
        "I'll be home late tonight, don't wait up for dinner.",
        
        # --- Spam (Fraudulent/Marketing Messages) ---
        "URGENT! You have won a 1-week free cruise prize! Call 09061701461 now to claim!",
        "FREE ringtones just text reply to this message to get yours now!",
        "Dear customer, your bank account has been locked. Click here to reset your PIN.",
        "Get cheap loans instantly with 0% interest! Limited time offer!",
        "CONGRATULATIONS! You have been selected for a cash prize. Claim within 24 hours.",
        "WINNER! As a valued network customer you have been selected to receive a £900 prize!",
        "Private Account Statement for your eyes only. Show entry code 4512 to collect reward.",
        "Double your income working from home! No experience needed. Sign up today.",
        "Your delivery failed. To reschedule your package, click the tracking link below.",
        "IMPORTANT NOTICE: Verify your account details immediately to avoid suspension."
    ],
    'label': [
        'ham', 'ham', 'ham', 'ham', 'ham', 'ham', 'ham', 'ham', 'ham', 'ham',
        'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam'
    ]
}

df = pd.DataFrame(data)

# 2. Split the dataset
X = df['message']
y = df['label']

# Stratify ensure both train and test splits get a healthy mix of spam and ham
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 3. Enhanced Text Vectorization
# 'sublinear_tf=True' applies sublinear scaling to word counts, 
# which prevents long messages from distorting the model's math.
vectorizer = TfidfVectorizer(stop_words='english', lowercase=True, sublinear_tf=True)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 4. Train the Model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# 5. Evaluate the Model
y_pred = model.predict(X_test_tfidf)
print("--- Model Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))
print("-" * 30 + "\n")

# 6. Prediction System
def predict_user_message():
    print("=== Spam Detector System ===")
    print("Type your message below to test if it's spam (or type 'exit' to quit):")
    
    while True:
        user_input = input("\nEnter message: ")
        if user_input.lower() == 'exit':
            print("Exiting system. Goodbye!")
            break
            
        if not user_input.strip():
            print("Please enter a valid message.")
            continue
            
        user_tfidf = vectorizer.transform([user_input])
        prediction = model.predict(user_tfidf)[0]
        
        if prediction == 'spam':
            print("🚨 Result: SPAM DETECTED!")
        else:
            print("✅ Result: Clean Message (Not Spam).")

if __name__ == "__main__":
    predict_user_message()