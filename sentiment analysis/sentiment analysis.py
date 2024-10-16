import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB


data = pd.read_csv('reviews.csv')

x = data['review']
y = data['sentiment']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

vectorizer = CountVectorizer()

X_train_vectorized = vectorizer.fit_transform(x_train)

X_test_vectorized = vectorizer.transform(x_test)

model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

y_pred = model.predict(X_test_vectorized)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

def predict_sentiment(review):
    vectorized_review = vectorizer.transform([review])
    prediction = model.predict(vectorized_review)
    return "Positive" if prediction[0] == 1 else "Negative"

new_review = [
    "this movie is next level! i loved it alot!",
    "terrible film , assal bale! waste of time and money.",
    "it was okay, not that good, and nopt that bad eaither."
]

for review in new_review:
    print(f"Review: {review}")
    print(f"Predicted sentiment: {predict_sentiment(review)}\n")

print ("Now can enter your own reviews!")
while True:
    user_review = input("enter a movie review ( or say 'quit' to stope it!):")
    if user_review.lower() == "quit":
        break
    print(f"Predicted sentiment: {predict_sentiment(user_review)}\n")
print(" thanks guys! jagratta! ede manam cheyyalsindi!")