import pickle
import pandas as pd

# Load model and encoders
model = pickle.load(open("model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

# Input match
input_data = {
    'team1': 'Mumbai Indians',
    'team2': 'Chennai Super Kings',
    'toss_winner': 'Mumbai Indians',
    'toss_decision': 'bat',
    'venue': 'Mumbai'
}

df = pd.DataFrame([input_data])

# Encode input
for col in df.columns:
    df[col] = encoders[col].transform(df[col])

# Prediction
prediction = model.predict(df)
probability = model.predict_proba(df)

# Decode winner
winner = encoders['winner'].inverse_transform(prediction)

# Confidence
confidence = max(probability[0]) * 100

print("Predicted Winner:", winner[0])
print("Confidence:", round(confidence, 2), "%")